import os
import pickle
from os.path import abspath, dirname, isfile, join
from typing import List, Tuple
from rlbench.tasks.pick_and_place import DatasetConfig

import numpy as np

CURRENT_DIR = dirname(abspath(__file__))


def get_pools(dir: str, keys: List[str]) -> Tuple[List[str], List[str], List[str]]:
    ttm_dict = {}
    for key in keys:
        ttm_dict[key] = [join(key, ttm[:-4]) for ttm
                         in os.listdir(join(dir, key))
                         if ttm[-4:] == '.ttm']
    target_pool = ttm_dict['targets']
    container_pool = ttm_dict['bowls'] + ttm_dict['boxes'] + ttm_dict['plates']
    distractor_pool = [item for sublist in ttm_dict.values()
                       for item in sublist]
    return target_pool, container_pool, distractor_pool


def separate_pool(pool: List[str], test_size: int) -> Tuple[List[str], List[str]]:
    test_pool = np.random.choice(pool, test_size, False)
    return list(set(pool).difference(set(test_pool))), test_pool


def get_objects(target_pool, container_pool, distractor_pool) -> List[str]:
    target = np.random.choice(target_pool)
    container = np.random.choice(container_pool)
    distractor_pool = [
        i for i in distractor_pool if i not in [target, container]]
    distractors = np.random.choice(distractor_pool, 2, False).tolist()
    return [target, container] + distractors


def get_texture_pools(dir: str, test_size: int) -> Tuple[List[str], List[str]]:
    textures = [join(txt[:-4]) for txt
                in os.listdir(dir)
                if txt[-4:] == '.png']
    test = np.random.choice(textures, test_size, False)
    return list(set(textures).difference(set(test))), test


def get_textures(texture_pool, size: int) -> List[str]:
    return np.random.choice(texture_pool, size, False).tolist()


def main():
    VARIATION = 500
    EPISODE = 5
    TEST_VARIATION = 100
    TEST_EPISODE = 2

    # object pool
    object_dir = os.path.join(CURRENT_DIR,
                              '../rlbench/assets/pick_and_place_ttms')
    keys = ['bowls', 'boxes', 'targets', 'plates']
    target_pool, container_pool, distractor_pool = get_pools(object_dir, keys)
    target_train, target_test = separate_pool(target_pool, 5)
    container_train, container_test = separate_pool(container_pool, 5)
    distractor_train, distractor_test = separate_pool(distractor_pool, 5)
    # print(target_pool, container_pool, distractor_pool, sep='\n')
    # print(len(target_pool), len(target_train), len(target_test))
    # print(len(distractor_pool), len(distractor_train), len(distractor_test))
    # print(len(container_pool), len(container_train), len(container_test))

    # texture pool
    texture_dir = os.path.join(CURRENT_DIR,
                               '../rlbench/assets/textures')
    texture_train, texture_test = get_texture_pools(texture_dir, 200)

    # training set
    train_objects = []
    train_textures = []
    for v in range(VARIATION):
        train_objects.append(get_objects(
            target_train, container_train, distractor_train))
        train_textures.append(get_textures(
            texture_train, len(train_objects[-1])))

    # testing set
    test_objects = []
    test_textures = []
    for v in range(TEST_VARIATION):
        test_objects.append(get_objects(
            target_test, container_test, distractor_test))
        test_textures.append(get_textures(
            texture_test, len(train_objects[-1])))

    # dump to pickle
    config_filename = os.path.join(CURRENT_DIR,
                                   '../rlbench/assets', 'dataset_config.pkl')
    config = DatasetConfig(VARIATION, EPISODE, TEST_VARIATION, TEST_EPISODE,
                           train_objects, train_textures, test_objects, test_textures)
    with open(config_filename, 'wb') as f:
        pickle.dump(config, f)

    with open(config_filename, 'rb') as f:
        _config = pickle.load(f)
        print(_config.VARIATION)
        print(_config.EPISODE)
        print(_config.TEST_VARIATION)
        print(_config.TEST_EPISODE)
        print(len(_config.train_objects), _config.train_objects[0])
        print(len(_config.train_textures), _config.train_textures[0])
        print(len(_config.test_objects), _config.test_objects[0])
        print(len(_config.test_textures), _config.test_textures[0])


if __name__ == '__main__':
    # app.run(main)
    main()
