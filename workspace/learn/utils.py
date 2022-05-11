import os
import pickle
from os.path import exists, join
from typing import Tuple

import numpy as np
import torch
from rlbench.backend.const import *


class BatchTensor():
    def __init__(self, demo, device=None) -> None:
        self.rgb = torch.tensor(demo['rgb'], device=device)
        self.depth = torch.tensor(demo['depth'], device=device)
        self.state = torch.tensor(demo['state'], device=device)
        self.action = torch.tensor(demo['action'], device=device)
        self.predict = torch.tensor(demo['predict'], device=device)

    def unpack(self):
        return self.rgb, self.depth, self.state, self.action, self.predict


class DatasetLoader():
    def __init__(self, dataset_root, task_name, batch_size: int, frames: int, debug=False) -> None:
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.batch_size = batch_size
        self.frames = frames
        self.debug = debug
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
        print('dataset size: variation:', variations,
              'episodes:', fixed_num_episode)

    def get_dataset_size(self) -> Tuple[int, int]:
        return self.variation_size, self.episode_size

    def load_demo(self, variation: int, episode: int):
        demo_pkl = join(self.dataset_root, self.task_name,
                        VARIATIONS_FOLDER % variation,
                        EPISODES_FOLDER, EPISODE_FOLDER % episode,
                        'demo.pkl')
        if self.debug:
            print('load_stored_demo: v', variation, 'e', episode)
        if not os.path.exists(demo_pkl):
            raise RuntimeError(
                'pkl of variation %d episodes %d not exist' % (variation, episode))
        demo = pickle.load(open(demo_pkl, 'rb'))
        return demo

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
            sampled_demo[key] = np.take(demo[key], fs, axis=0)
            # for i, f in enumerate(fs):
            #     assert np.array_equal(sampled_demo[key][i], demo[key][f])
        return sampled_demo

    def get_sampled_demo(self, variation: int, episode: int,
                         rng: np.random.Generator):
        return self.sample_demo(self.load_demo(
            variation % self.variation_size, episode % self.episode_size), rng)

    def get_batch_sampled_demo(self, iterration: int):
        """iterration as random seed!"""
        rng = np.random.default_rng(iterration)
        v_idx = iterration * self.batch_size
        e_idx = v_idx // self.variation_size
        if self.debug:
            print('batch: i %d v %d e %d' % (iterration, v_idx, e_idx))
        demos = [
            self.get_sampled_demo(v, e_idx, rng)
            for v in range(v_idx, v_idx + self.batch_size)
        ]
        batch_demo = {}
        for k in demos[0]:
            batch_demo[k] = np.stack([d[k] for d in demos], axis=0)
        return batch_demo

    def get_batch_sampled_demo_pair(self, iterration: int):
        """iterration as random seed!
        secure: not to sample same episode within the same variation
        """
        def gen_e_idx():
            x = rng.integers(0, self.episode_size - 1)
            return x if x < e_idx else (x + 1) % self.episode_size

        rng = np.random.default_rng(iterration)
        v_idx = iterration * self.batch_size
        e_idx = v_idx // self.variation_size
        if self.debug:
            print('batch pair: i %d v %d e %d' % (iterration, v_idx, e_idx))
        r_demos = [
            self.get_sampled_demo(v, e_idx, rng)
            for v in range(v_idx, v_idx + self.batch_size)
        ]
        h_demos = [
            self.get_sampled_demo(v, gen_e_idx(), rng)
            for v in range(v_idx, v_idx + self.batch_size)
        ]
        h_batch_demo, r_batch_demo = {}, {}
        for k in h_demos[0]:
            h_batch_demo[k] = np.stack([d[k] for d in h_demos], axis=0)
        for k in r_demos[0]:
            r_batch_demo[k] = np.stack([d[k] for d in r_demos], axis=0)
        return h_batch_demo, r_batch_demo


if __name__ == '__main__':
    """testing"""
    dataset_root = join(os.path.expanduser('~'), 'disk/dataset')
    task_name = 'pick_and_place'
    batch_size = 4
    frames = 50
    dataloader = DatasetLoader(
        dataset_root, task_name, batch_size, frames, True)
    # batch_demo = dataloader.get_batch_sampled_demo(0)
    h_batch, r_batch = dataloader.get_batch_sampled_demo_pair(0)
    for k, v in h_batch.items():
        print(k, v.shape)
    for k, v in r_batch.items():
        print(k, v.shape)
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )  # set device
    h_rgb, h_depth, h_state, h_action, h_predict = BatchTensor(h_batch, device).unpack()
    r_rgb, r_depth, r_state, r_action, r_predict = BatchTensor(r_batch, device).unpack()
    print(h_rgb.shape, h_depth.shape, h_state.shape, h_action.shape, h_predict.shape)
    print(r_rgb.shape, r_depth.shape, r_state.shape, r_action.shape, r_predict.shape)
