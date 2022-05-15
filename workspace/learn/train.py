from collections import OrderedDict

import torch
from absl import app
from torch.utils.data import DataLoader
from tqdm import tqdm
from workspace.learn.data import PairDataset
from workspace.learn.flags import FLAGS
from workspace.models.daml import Daml, save_model
from workspace.utils import check_and_make, get_logger


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

    logger.info('train with flags: %s' % str(FLAGS.flag_values_dict()))

    torch.backends.cudnn.benchmark = True

    dataset = PairDataset(dataset_root, task_name, T, dataset_seed, True)
    dataloader = DataLoader(
        dataset, meta_batch_size, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
    dataset_len = len(dataset)
    num_dataset_batch = (dataset_len // meta_batch_size) + \
        (1 if dataset_len % meta_batch_size else 0)
    print('num_dataset_batch', num_dataset_batch)

    model = Daml()
    meta_optimizer = torch.optim.Adam(model.parameters())

    epoch = 0
    adapt_running_loss = torch.zeros(
        num_updates, device=model.device, requires_grad=False)
    meta_running_loss = torch.zeros(
        3, device=model.device, requires_grad=False)

    for i in tqdm(range(iteration)):
        model.zero_grad()  # remember to zero grad

        if i % save_iter == 0:
            # save model
            save_model(model, meta_optimizer, i)
            logger.info('%d model-saved' % i)
        if i % log_iter == 0:
            # log mean adapt-loss
            logger.info('%d adapt-loss %s' % (
                i, (adapt_running_loss / log_iter).tolist()))
            adapt_running_loss.zero_()
            # log mean meta-loss
            logger.info('%d meta-loss %s' % (
                i, (meta_running_loss / log_iter).tolist()))
            meta_running_loss.zero_()

        # get a batch of demo pair
        if i % num_dataset_batch == 0:
            loader = iter(dataloader)
            logger.info('%d new-epoch %d' % (i, epoch))
            epoch += 1

        r_d, h_d = next(loader)
        r_rgb, r_depth, r_state, r_action, r_predict = [
            f.to(model.device, non_blocking=True) for f in r_d]
        h_rgb, h_depth, _, _, _ = [
            f.to(model.device, non_blocking=True) for f in h_d]

        # adapt
        pre_update = OrderedDict(
            model.meta_named_parameters())  # pre-update params
        batch_post_update, adapt_losses = model.adapt(
            h_rgb, h_depth, pre_update, adapt_lr, num_updates
        )  # adaptation (get post-update params)
        adapt_running_loss.add_(adapt_losses)

        # meta
        meta_loss, meta_loss_mat = model.meta_loss(
            r_rgb, r_depth, r_state, r_action, r_predict, batch_post_update
        )  # post-update
        meta_running_loss.add_(meta_loss_mat)

        meta_loss.backward()
        meta_optimizer.step()

    # save final
    # save model
    model.zero_grad()
    save_model(model, meta_optimizer, iteration)
    logger.info('%d model-saved' % iteration)
    # log mean adapt-loss
    logger.info('%d adapt-loss %s' % (
        iteration, (adapt_running_loss / log_iter).tolist()))
    # log mean meta-loss
    logger.info('%d meta-loss %s' % (
        iteration, (meta_running_loss / log_iter).tolist()))
    logger.info('training done, iteration %d epoch %d' % (iteration, epoch))


if __name__ == '__main__':
    app.run(main)
