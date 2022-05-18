import os
import pickle
from collections import OrderedDict
from os.path import exists, join

import numpy as np
import torch
from absl import app
from PIL import Image
from rlbench.backend.const import *
from torch.utils.data import DataLoader, Dataset, Sampler
from workspace.learn.flags import FLAGS


class SequentialDataset(Dataset):
    def __init__(self, dataset_root, task_name, frames, seed=None, debug=False) -> None:
        super().__init__()
        self.debug = debug
        self.rng = np.random.default_rng(seed)
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.frames = frames
        # get variation_size & episode_size
        task_root = join(dataset_root, task_name)
        if not exists(task_root):
            raise RuntimeError("Can't find the dataset for %s at: %s" % (
                task_name, task_root))
        fixed_num_episode = -1
        variations = len(os.listdir(task_root))
        for v in range(variations):
            episodes_root = join(task_root, VARIATIONS_FOLDER %
                                 v, EPISODES_FOLDER)
            episodes = len(os.listdir(episodes_root))
            if fixed_num_episode < 0:
                fixed_num_episode = episodes
                if self.debug:
                    print('expect fixed number of episodes: %d' %
                          fixed_num_episode)
            if episodes != fixed_num_episode:
                raise RuntimeError(
                    'variation %d has different number of episodes %d' % (v, episodes))
            for e in range(episodes):
                episode_path = join(episodes_root, EPISODE_FOLDER % e)
                if not (os.path.exists(episode_path)
                        and os.path.exists(join(episode_path, 'demo.pkl'))
                        and os.path.exists(join(episode_path, 'ref.gif'))):
                    raise RuntimeError(
                        'variation %d episodes %d not exist' % (v, episodes))
        self.variation_size = variations
        self.episode_size = fixed_num_episode
        print('dataset_root %s \n'
              'task_name: %s variations: %d episodes: %d frames: %d' % (
                  dataset_root, task_name, variations, fixed_num_episode, frames
              ))

    def __len__(self):
        return self.variation_size * self.episode_size

    def __getitem__(self, idx):
        e, v = divmod(idx, self.variation_size)
        d = self.load_demo(v, e)
        d = self.sample_demo(d, self.rng)
        return d['rgb'] / 255.0, d['depth'], d['state'], d['action'], d['predict']

    def get_v_e(self, idx):
        e, v = divmod(idx, self.variation_size)
        return v, e

    def load_demo(self, variation: int, episode: int):
        demo_pkl = join(self.dataset_root, self.task_name,
                        VARIATIONS_FOLDER % variation,
                        EPISODES_FOLDER, EPISODE_FOLDER % episode,
                        'demo.pkl')
        if self.debug:
            print('load_demo: v', variation, 'e', episode)
        if not os.path.exists(demo_pkl):
            raise RuntimeError(
                'pkl of variation %d episodes %d not exist' % (variation, episode))
        return pickle.load(open(demo_pkl, 'rb'))

    def sample_demo(self, demo, rng: np.random.Generator):
        """frames: how many frames to sample
        every frame should be sampled at least once if frames > l
        """
        l = demo['rgb'].shape[0]
        fs = sorted(
            rng.choice(l, self.frames, replace=False) if self.frames <= l
            else np.concatenate((np.arange(l), rng.choice(l, self.frames - l)))
        )
        sampled_demo = {}
        for key in demo:
            sampled_demo[key] = np.take(
                demo[key], fs, axis=0).astype(np.float32)
            # for i, f in enumerate(fs):
            #     assert np.array_equal(sampled_demo[key][i], demo[key][f])
        # imgs = [
        #     Image.fromarray(arr.transpose(1, 2, 0))
        #     for arr in sampled_demo['rgb']
        # ]
        # imgs[0].save('tmp.gif', save_all=True,
        #              append_images=imgs[1:], duration=100, loop=0)
        # print(len(imgs))
        # print(sampled_demo['state'].shape, sampled_demo['state'][0], sep='\n')
        if not FLAGS.gripper_action:  # remove gripper_open, help gripper close
            sampled_demo['state'] = np.take(
                sampled_demo['state'], [0, 1, 2, 3, 4, 5, 7, 8, 9], axis=1)
        if FLAGS.gripper_action:  # use fixed gripper action
            for action, gripper in zip(sampled_demo['action'], sampled_demo['gripper_action']):
                action[-1] = gripper[0]
        # print(sampled_demo['state'].shape, sampled_demo['state'][0], sep='\n')
        return sampled_demo


class PairDataset(SequentialDataset):
    def __init__(self, dataset_root, task_name, frames, seed=None, exclude=False, debug=False) -> None:
        super().__init__(dataset_root, task_name, frames, seed, debug)
        self.exclude = exclude  # exclude episode in corresponding vanilla dataset
        self.episode_rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        e, v = divmod(idx, self.variation_size)
        r_d = self.load_demo(v, e)
        r_d = self.sample_demo(r_d, self.rng)
        r_d = (r_d['rgb'] / 255.0, r_d['depth'],
               r_d['state'], r_d['action'], r_d['predict'])
        e += self.episode_rng.integers(1 if self.exclude else 0,
                                       self.episode_size)
        e = e % self.episode_size
        h_d = self.load_demo(v, e)
        h_d = self.sample_demo(h_d, self.episode_rng)
        h_d = (h_d['rgb'] / 255.0, h_d['depth'],
               h_d['state'], h_d['action'], h_d['predict'])
        return r_d, h_d


def test_dataset(dataset: SequentialDataset):
    print('----- test_dataset -----')
    demos = [dataset[i] for i in range(0, 2)]
    # demos = [dataset[i] for i in range(500, 510)]
    # for v in demos[0]:
    #     print(v.shape)
    # print(demos[0][0][-1])
    # print((demos[0][0] * 255)[-1])


def test_dataloader(dataset: SequentialDataset, randdataset: PairDataset):
    print('----- test_dataloader -----')
    batch_size = 4
    r_dataloader = DataLoader(
        dataset, batch_size, pin_memory=True, num_workers=0, persistent_workers=False)
    h_dataloader = DataLoader(
        randdataset, batch_size, pin_memory=True,  num_workers=0, persistent_workers=False, shuffle=True)
    r_loader = iter(r_dataloader)
    for _ in range(10):
        for v in next(r_loader):
            print(v.is_pinned(), v.shape)
    h_loader = iter(h_dataloader)
    print('----- paired -----')
    for _ in range(10):
        for p in next(h_loader):
            for v in p:
                print(v.is_pinned(), v.shape)


def main(argv):
    dataset_root = FLAGS.dataset_root
    task_name = FLAGS.task_name
    meta_batch_size = FLAGS.meta_batch_size
    T = FLAGS.T
    dataset = SequentialDataset(dataset_root, task_name, T, 0, True)
    test_dataset(dataset)
    randdataset = PairDataset(dataset_root, task_name, T, 0, True, True)
    # test_dataloader(dataset, randdataset)


if __name__ == '__main__':
    app.run(main)
