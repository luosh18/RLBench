from collections import OrderedDict
from absl import app
from workspace.learn.data import RandomDataset, SequentialDataset
from workspace.learn.flags import FLAGS
from workspace.models.daml import Daml, save_model
from workspace.utils import check_and_make, get_logger

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(argv):
    save_dir = FLAGS.save_dir
    if check_and_make(FLAGS.save_dir):
        print('save dir already exist!')
        return
    logger = get_logger('train')
    iteration = FLAGS.iteration
    save_iter = FLAGS.save_iter
    log_iter = FLAGS.log_iter
    dataset_root = FLAGS.dataset_root
    task_name = FLAGS.task_name
    dataset_seed = FLAGS.dataset_seed
    T = FLAGS.T
    meta_batch_size = FLAGS.meta_batch_size
    adapt_lr = FLAGS.adapt_lr
    num_updates = FLAGS.num_updates

    r_dataset = SequentialDataset(dataset_root, task_name, T, dataset_seed)
    h_dataset = RandomDataset(dataset_root, task_name, T, dataset_seed, True)
    r_dataloader = DataLoader(r_dataset, meta_batch_size, pin_memory=True)
    h_dataloader = DataLoader(h_dataset, meta_batch_size, pin_memory=True)
    r_loader = iter(r_dataloader)
    h_loader = iter(h_dataloader)

    model = Daml()
    meta_optimizer = torch.optim.Adam(model.parameters())

    adapt_running_loss = torch.zeros(num_updates, device=model.device)
    meta_running_loss = torch.zeros(3, device=model.device)

    for i in tqdm(range(iteration)):
        model.zero_grad()  # remember to zero grad

        if i % save_iter == 0:
            # save model
            save_model(model, meta_optimizer, str(i))
            logger.info('model saved @ %d' % i)
        if i % log_iter == 0:
            # log mean adapt-loss
            logger.info('adapt-loss @ %d %s' % (
                i, (adapt_running_loss / log_iter).tolist()))
            adapt_running_loss.zero_()
            # log mean meta-loss
            logger.info('meta-loss @%d %s' % (
                i, (meta_running_loss / log_iter).tolist()))
            meta_running_loss.zero_()

        # get a batch of demo pair
        try:
            r_rgb, r_depth, r_state, r_action, r_predict = [
                f.to(model.device) for f in next(r_loader)]
            h_rgb, h_depth, _, _, _ = [
                f.to(model.device) for f in next(h_loader)]
        except StopIteration:
            logger.info('new epoch @ %d' % i)
            r_loader = iter(r_dataloader)
            h_loader = iter(h_dataloader)
            r_rgb, r_depth, r_state, r_action, r_predict = [
                f.to(model.device) for f in next(r_loader)]
            h_rgb, h_depth, _, _, _ = [
                f.to(model.device) for f in next(h_loader)]

        # adapt
        pre_update = OrderedDict(
            model.meta_named_parameters())  # pre-update params
        batch_post_update, adapt_losses = model.adapt(
            h_rgb, h_depth, pre_update, adapt_lr, num_updates
        )  # adaptation (get post-update params)
        adapt_running_loss += adapt_losses

        # meta
        meta_loss, meta_loss_mat = model.meta_loss(
            r_rgb, r_depth, r_state, r_action, r_predict, batch_post_update
        )  # post-update
        meta_running_loss += meta_loss_mat

        meta_loss.backward()
        meta_optimizer.step()


if __name__ == '__main__':
    app.run(main)
