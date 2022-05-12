from workspace.learn.data import SequentialDataset, RandomDataset
from workspace.models.daml import Daml

from absl import app, flags

FLAGS = flags.FLAGS


def main(argv):
    print('im_height', FLAGS.im_height)


if __name__ == '__main__':
    app.run(main)
