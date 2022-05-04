from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchsummary
from absl import app, flags

from torch_utils import *
from utils import Timer

FLAGS = flags.FLAGS

# Dataset/Method options
flags.DEFINE_integer('im_width', 128,
                     'width of the images in the demo videos -- 125 for sim_push, 128 to suit RLBench')
flags.DEFINE_integer('im_height', 128,
                     'height of the images in the demo videos -- 125 for sim_push, 128 to suit RLBench')
flags.DEFINE_integer('num_channels', 3,
                     'number of channels of the images in the demo videos')

# Training Options
flags.DEFINE_bool('cuda', True, 'using GPU, default true')
flags.DEFINE_integer('num_eept', 2,
                     'number of predictions of end-effector (gripper) position'
                     ' -- 2 (target & container) for p&p')
flags.DEFINE_integer('eept_dim', 3,
                     'dimension of end-effector position')
flags.DEFINE_bool('conv_bt', True,
                  'use bias transformation for the first conv layer, N/A for using pretraining')
flags.DEFINE_integer('bt_dim', 20,
                     'the dimension of bias transformation for FC layers -- 20 for all exp')

# Model Options
flags.DEFINE_bool('fp', True, 'use spatial soft-argmax or not')  # frozen
# flags.DEFINE_integer('num_conv_layers', 5,
#                      'number of conv layers -- 5 for placing')
flags.DEFINE_integer('num_filters', 64,
                     'number of filters for conv nets -- 64 for placing')
flags.DEFINE_integer('filter_size', 3,
                     'filter size for conv nets -- 3 for placing')
# flags.DEFINE_string('initialization', 'xavier',
#                     'initializer for conv weights. Choose among random, xavier, and he')
flags.DEFINE_list('strides', [2, 2, 2, 1, 1],
                  'stride of conv layers -- p&p: [2, 2, 2, 1, 1]')
flags.DEFINE_integer('temporal_filter_size', 10,
                     'filter size for temporal convolution -- 10x1 for p&p')
flags.DEFINE_list('num_temporal_filters', [10, 30, 30],
                  'number of filters for temporal convolution for ee pose prediction')
flags.DEFINE_integer('num_fc_layers', 4,
                     'number of fully-connected layers -- 4 for p&p')
flags.DEFINE_integer('fc_layer_size', 50,
                     'hidden dimension of fully-connected layers -- 50 for p&p')


class Daml(nn.Module):
    def __init__(self, action_size=7, state_size=7, img_idx=None) -> None:
        super().__init__()
        self.action_size = action_size  # dim action (output)
        self.state_size = state_size  # dim robot config (state)
        # build model
        with Timer("Building pytorch network"):
            self.construct_model()

    def construct_model(self):
        # build model
        # VGG CNN layers
        num_filters = FLAGS.num_filters  # pushing default 64
        filter_size = FLAGS.filter_size  # pushing default 3

        # reshape input image
        im_width = FLAGS.im_width
        im_height = FLAGS.im_height
        num_channels = FLAGS.num_channels

        # weights initialization xavier for pushing, TODO: add more initialization option
        # initialization = FLAGS.initialization

        # input channel (plus bias transformation channel)
        fan_in = num_channels * (2 if FLAGS.conv_bt else 1)

        # init bias transformation, add to parameters and clip it to [0.0,1.0]
        self.conv_bt = nn.Parameter(
            torch.clamp(
                torch.zeros(
                    [num_channels, im_height, im_width],
                    dtype=torch.float32,
                    requires_grad=True,
                ),
                min=0.0,
                max=1.0,
            )
        ) if FLAGS.conv_bt else None

        ###### 2d convolution layers ######
        # p&p: the first three layers are with stride 2 and the last two layer is with stride 1
        strides = list(map(int, FLAGS.strides))
        # TODO: fix padding. MIL use padding='same' in tf,
        # pytorch: padding='same' is not supported for strided convolutions
        self.convs = [
            nn.Conv2d(fan_in, num_filters, filter_size,
                      strides[0], padding='same' if strides[0] == 1 else 0)
        ] + [
            nn.Conv2d(num_filters, num_filters, filter_size,
                      stride, padding='same' if stride == 1 else 0)
            for stride in strides[1:]
        ]
        for conv in self.convs:
            # FLAGS.initialization == 'xavier'. init weight and bias
            init_weights_xavier(conv.weight)
            init_weights_zeros(conv.bias)

        # FLAGS.fp == True. spatial softmax after 2d cnv, equals to the twice size of filters (checked)
        self.n_conv_output = num_filters * 2

        ###### temporal convolution layers: predicted final eept, upper head ######
        ###### ??? the output state should be corresponding to the provided data ######
        temporal_filter_size = FLAGS.temporal_filter_size  # tcn filter size
        n_temporal_filters = list(map(int, FLAGS.num_temporal_filters))
        # TODO: assert size of conv output & temp_convs_ee input
        self.temp_convs_ees = [
            [  # set input channels to the size of conv output
                nn.Conv1d(self.n_conv_output, n_temporal_filters[0], temporal_filter_size,
                          padding='same')
            ] + [
                nn.Conv1d(n_temporal_filters[i-1], n_temporal_filters[i], temporal_filter_size,
                          padding='same')
                for i in range(1, len(n_temporal_filters))
            ] + [  # final output should be the predicted gripper pose
                nn.Conv1d(n_temporal_filters[-1], FLAGS.eept_dim, 1,
                          padding='same')
            ]
            for _ in range(FLAGS.num_eept)
        ]

        for temp_convs_ee in self.temp_convs_ees:
            for temp_conv_ee in temp_convs_ee:
                init_weights_normal(temp_conv_ee.weight, std=0.01)
                init_weights_zeros(temp_conv_ee.bias)

        ###### fully-connected layers preparation ######
        # after spatial softmax 2*channel concat predicted gripper pose
        fc_shape_in = self.n_conv_output + FLAGS.num_eept * FLAGS.eept_dim
        # concat state vector
        fc_shape_in += self.state_size
        # input concat bias transformation
        fc_shape_in += FLAGS.bt_dim
        # build weight for fc bias transformation
        self.fc_bt = nn.Parameter(
            torch.zeros(FLAGS.bt_dim, dtype=torch.float32, requires_grad=True)
        ) if FLAGS.bt_dim > 0 else None

        ####### build fully-connected layers #######
        n_fc_layer = FLAGS.num_fc_layers   # default 4 for p&p daml
        fc_layer_size = FLAGS.fc_layer_size  # default 50 for p&p daml
        self.fc = [
            [
                nn.Linear(fc_shape_in, fc_layer_size)
            ] + [
                nn.Linear(fc_layer_size, fc_layer_size)
                for _ in range(n_fc_layer - 2)
            ] + [
                nn.Linear(fc_layer_size, self.action_size)
            ]
        ]
        # ?? double the use of last 2 layers, output is action

        ###### temporal convolution layers: adaptation loss, upper head ######
        self.temp_convs = [
            # set input channels to the size of fc output
            nn.Conv1d(fc_layer_size, n_temporal_filters[0],
                      temporal_filter_size, padding='same')
        ] + [
            nn.Conv1d(n_temporal_filters[i-1], n_temporal_filters[i],
                      temporal_filter_size, padding='same')
            for i in range(1, len(n_temporal_filters))
        ] + [  # final output should be the predicted gripper pose
            nn.Conv1d(n_temporal_filters[-1], self.action_size,
                      1, padding='same')
        ]

        for temp_conv in self.temp_convs:
            init_weights_normal(temp_conv.weight, std=0.01)
            init_weights_zeros(temp_conv.bias)

    def forward_conv(self, image_input, testing=False):
        pass


def main(argv):
    # set device
    device = torch.device(
        'cuda' if FLAGS.cuda and torch.cuda.is_available() else 'cpu'
    )

    im_width = FLAGS.im_width
    im_height = FLAGS.im_height
    num_channels = FLAGS.num_channels
    fan_in = num_channels * (2 if FLAGS.conv_bt else 1)
    data = torch.rand([num_channels, im_height, im_width])
    model = Daml(7, 7, im_height * im_width)
    print(model)
    model.cuda()
    torchsummary.summary(model, data.shape)


if __name__ == '__main__':
    app.run(main)
