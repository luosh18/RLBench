from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torchsummary
from absl import app, flags
from torchmeta.modules import (MetaConv1d, MetaConv2d, MetaLinear, MetaModule,
                               MetaSequential)
from torchmeta.utils.gradient_based import gradient_update_parameters

import mdn
from meta_groupnorm import MetaGroupNorm
from normalize import Normalize
from saptial_softmax import SpatialSoftmax
from utils import Timer

FLAGS = flags.FLAGS

# Dataset/Method options
flags.DEFINE_integer('im_width', 128,
                     'width of the images in the demo videos -- 125 for sim_push, 128 to suit RLBench')
flags.DEFINE_integer('im_height', 128,
                     'height of the images in the demo videos -- 125 for sim_push, 128 to suit RLBench')
flags.DEFINE_integer('num_channels', 3,
                     'number of channels of the images in the demo videos')
flags.DEFINE_integer('state_size', 7,
                     'dimension of robot state -- 7 for Jaco (include gripper)')
flags.DEFINE_integer('action_size', 7,
                     'dimension of robot action -- 7 for Jaco (include gripper)')
flags.DEFINE_integer('T', 50,
                     'time horizon of the demo videos -- 50 for reach, 100 for push, 50 for DAML')
flags.DEFINE_integer('mdn_samples', 100,
                     'sample "mdn_samples" actions from MDN and choose the one with highest probability')

# Training Options
flags.DEFINE_bool('cuda', True, 'using GPU, default true')

## Model Options ##
# 2D Convs
flags.DEFINE_list('strides', [2, 2, 2, 1, 1],
                  'list of 2d conv stride, len is num of conv layers -- p&p: [2, 2, 2, 1, 1]')
flags.DEFINE_integer('num_filters', 64,
                     'number of filters for conv nets -- 64 for placing')
flags.DEFINE_integer('filter_size', 3,
                     'filter size for conv nets -- 3 for placing')
flags.DEFINE_integer('num_depth_filters', 16,
                     'number of filters for depth conv layer -- 16 for placing')
flags.DEFINE_bool('conv_bt', True,
                  'use bias transformation for the first conv layer, N/A for using pretraining')
# Gripper Pose Prediction
flags.DEFINE_integer('eept_dim', 3,
                     'dimension of end-effector position')
# 1D Temporal Convs
flags.DEFINE_list('num_temp_filters', [10, 30, 30],
                  'number of filters for temporal convolution for ee pose prediction')
flags.DEFINE_integer('temp_filter_size', 10,
                     'filter size for temporal convolution -- 10x1 for p&p')
# full-connected layers
flags.DEFINE_integer('num_fc_layers', 4,
                     'number of fully-connected layers -- 4 for p&p')
flags.DEFINE_integer('fc_layer_size', 50,
                     'hidden dimension of fully-connected layers -- 50 for p&p')
flags.DEFINE_integer('bt_dim', 20,
                     'the dimension of bias transformation for FC layers -- 20 for all exp')
flags.DEFINE_integer('num_gaussian', 20,
                     'number of Gaussian kernels in MDN')


class Daml(MetaModule):
    def __init__(self) -> None:
        super().__init__()
        self.state_size = FLAGS.state_size
        self.action_size = FLAGS.action_size
        self.device = torch.device(
            'cuda' if FLAGS.cuda and torch.cuda.is_available() else 'cpu'
        )  # set device
        with Timer('contruct DAML model'):
            self.construct_model()
            self.init_weights()
            self.to(self.device)

    def construct_model(self):
        num_filters = FLAGS.num_filters
        filter_size = FLAGS.filter_size
        num_channels = FLAGS.num_channels

        ###### 2d convolution layers ######
        # p&p: the first three layers are with stride 2 and the last two layer is with stride 1
        strides = list(map(int, FLAGS.strides))
        # TODO: fix padding. MIL use padding='same' in tf,
        # pytorch: padding='same' is not supported for stride != 1 convolutions

        ##### RGB first layer #####
        # input channel (plus bias transformation channel)
        rgb_in = num_channels * (2 if FLAGS.conv_bt else 1)
        self.rgb_conv = MetaSequential(
            MetaConv2d(rgb_in, num_filters, filter_size,
                       strides[0], padding='same' if strides[0] == 1 else strides[0]),
            MetaGroupNorm(1, num_filters),
            # each conv2d should followed by layer normalization
            # use Group Norm instead, according to:
            # https://arxiv.org/pdf/1803.08494.pdf page:4 "3.1. Formulation" "Relation to Prior Work"
            nn.ReLU(),
        )

        ##### Depth first layer #####
        num_depth_filters = FLAGS.num_depth_filters
        depth_in = 2 if FLAGS.conv_bt else 1
        self.depth_conv = MetaSequential(
            MetaConv2d(depth_in, num_depth_filters, filter_size,
                       strides[0], padding='same' if strides[0] == 1 else strides[0]),
            MetaGroupNorm(1, num_depth_filters),
            nn.ReLU(),
        )

        ##### other conv layers #####
        convs = [
            MetaConv2d(num_filters + num_depth_filters, num_filters, filter_size,
                       strides[1], padding='same' if strides[1] == 1 else strides[1]),
        ]
        for stride in strides[2:]:
            convs.extend([
                MetaGroupNorm(1, num_filters),
                nn.ReLU(),
                MetaConv2d(num_filters, num_filters, filter_size,
                           stride, padding='same' if stride == 1 else stride),
            ])
        self.convs = MetaSequential(*convs)

        ###### Spatial Softmax after 2d conv ######
        self.spatial_softmax = SpatialSoftmax()
        # output shape equals to twice the size of filters (checked)
        self.num_feature_output = num_filters * 2

        ###### predict gipper pose ######
        # predict the pose of the gripper when it contacts the target object and/or container
        self.predict_pose = MetaLinear(self.num_feature_output, FLAGS.eept_dim)

        ###### fully-connected layers ######
        # input: cat: feature (spatial softmax), predicted_pose, robot state, bias transform
        fc_shape_in = self.num_feature_output + \
            FLAGS.eept_dim + self.state_size + FLAGS.bt_dim
        n_fc_layer = FLAGS.num_fc_layers   # default 4 for p&p daml
        fc_layer_size = FLAGS.fc_layer_size  # default 50 for p&p daml
        fcs = [
            MetaLinear(fc_shape_in, fc_layer_size),
        ]
        for _ in range(1, n_fc_layer):
            fcs.extend([
                nn.ReLU(),
                MetaLinear(fc_layer_size, fc_layer_size),
            ])
        self.fcs = MetaSequential(*fcs)

        ###### Action ######
        # Mixture Density Networks for continuous action [:6] (joint velocity)
        self.mdn = mdn.MDN(
            fc_layer_size, self.action_size - 1, FLAGS.num_gaussian)
        # Linear with Sigmoid for discrete action [6:] (gripper open/close)
        self.discrete = MetaSequential(
            MetaLinear(fc_layer_size, 1),
            nn.Sigmoid(),
        )
        ###### Temporal Convs ######

        def _construct(in_channels):
            num_temp_filters = list(map(int, FLAGS.num_temp_filters))
            temp_filter_size = FLAGS.temp_filter_size
            temp_convs = [
                MetaConv1d(in_channels, num_temp_filters[0], temp_filter_size,
                           padding='same'),
                MetaGroupNorm(1, num_temp_filters[0]),
                nn.ReLU(),
            ]
            for i, num_filter in enumerate(num_temp_filters[1:]):
                pre_num_filter = num_temp_filters[i]
                temp_convs.extend([
                    MetaConv1d(pre_num_filter, num_filter, temp_filter_size,
                               padding='same'),
                    MetaGroupNorm(1, temp_filter_size),
                    nn.ReLU(),
                ])
            temp_convs.extend([
                MetaConv1d(num_temp_filters[-1], 1, 1, padding='same'),
                Normalize(),  # perform L2 norm after convs (adaptive loss)
            ])
            return MetaSequential(*temp_convs)
        # temp conv on predict pose
        self.feature_temp_convs = _construct(self.num_feature_output)
        # temp conv on fc layers
        self.fc_temp_convs = _construct(FLAGS.fc_layer_size)

    def init_weights(self):
        im_height = FLAGS.im_height
        im_width = FLAGS.im_width
        num_channels = FLAGS.num_channels

        ###### Bias Transformation (meta-param) ######
        # init bias transformation, add to parameters and clip it to [0.0,1.0]
        self.rgb_bt = nn.Parameter(torch.clamp(  # TODO: meta-params
            torch.zeros(
                [num_channels, im_height, im_width], requires_grad=True
            ),
            min=0.0, max=1.0,
        )) if FLAGS.conv_bt else None
        self.depth_bt = nn.Parameter(torch.clamp(  # TODO: meta-params
            torch.zeros(
                [1, im_height, im_width], requires_grad=True
            ),
            min=0.0, max=1.0,
        )) if FLAGS.conv_bt else None
        # build weight for fc bias transformation
        self.fc_bt = nn.Parameter(torch.clamp(
            torch.zeros(FLAGS.bt_dim, requires_grad=True),
            min=0.0, max=1.0,
        )) if FLAGS.bt_dim > 0 else None

        for m in [
            # 2d convs
            *self.rgb_conv.children(),
            *self.depth_conv.children(),
            *self.convs.children(),
            # 1d convs (temp)
            *self.feature_temp_convs.children(),
            *self.fc_temp_convs.children(),
        ]:
            if isinstance(m, MetaConv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, MetaGroupNorm) or isinstance(m, MetaConv1d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ReLU) or isinstance(m, Normalize):
                pass
            else:
                raise Exception(
                    'module %s in convs not initialized' % m._get_name())
        # spatial softmax has no params
        # predict pose
        nn.init.normal_(self.predict_pose.weight, std=0.01)
        nn.init.zeros_(self.predict_pose.bias)
        # fc
        for m in self.fcs.children():
            if isinstance(m, MetaLinear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ReLU):
                pass
            else:
                raise Exception(
                    'module %s in fcs not initialized' % m._get_name())
        # action (MDN & discrete)
        # TODO: MDN for action don't need to be meta-params ?
        for m in [*self.mdn.children(), *self.discrete.children()]:
            if isinstance(m, MetaLinear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_conv(self, rgb_in, depth_in, params=None) -> torch.Tensor:
        # build bias transformation for conv2d
        rgb_bt = torch.zeros_like(rgb_in, device=self.device)
        rgb_bt += self.rgb_bt if params is None else params['rgb_bt']
        depth_bt = torch.zeros_like(depth_in, device=self.device)
        depth_bt += self.depth_bt if params is None else params['depth_bt']
        # concat image input and bias transformation
        rgb_in = torch.cat((rgb_in, rgb_bt), dim=1)
        depth_in = torch.cat((depth_in, depth_bt), dim=1)
        ###### CNN Forward ######
        rgb_out = self.rgb_conv(
            rgb_in, params=self.get_subdict(params, 'rgb_conv'))
        depth_out = self.depth_conv(
            depth_in, params=self.get_subdict(params, 'depth_conv'))
        # concat rgb & d channel wise
        conv_in = torch.cat((rgb_out, depth_out), dim=1)
        conv_out = self.convs(
            conv_in, params=self.get_subdict(params, 'convs'))
        ###### Spatial Softmax ######
        conv_out_features = self.spatial_softmax(conv_out)
        return conv_out_features

    def forward_predict_pose(self, conv_out, params=None) -> torch.Tensor:
        # outer meta-objective
        # predict the pose of the gripper when it contacts
        # the target object and/or container by linear (V. A.)
        return self.predict_pose(conv_out, self.get_subdict(params, 'predict_pose'))

    def forward_fc(self, conv_out, predict_pose, state_in=None, params=None) -> torch.Tensor:
        if state_in is None:
            state_in = torch.zeros(
                (conv_out.shape[0], self.state_size), device=self.device)
        # feature (spatial softmax), predicted_pose, robot state, bias transform
        fc_bt = torch.zeros(
            (conv_out.shape[0], *self.fc_bt.shape), device=self.device)
        fc_bt += self.fc_bt if params is None else params['fc_bt']
        fc_in = torch.cat([conv_out, predict_pose, state_in, fc_bt], dim=1)
        fc_out = self.fcs(fc_in, self.get_subdict(params, 'fcs'))
        return fc_out

    def forward_action(self, fc_out, params=None):  # MDN output
        pi, sigma, mu = self.mdn(
            fc_out, params=self.get_subdict(params, 'mdn'))
        discrete = self.discrete(
            fc_out, params=self.get_subdict(params, 'discrete'))
        return pi, sigma, mu, discrete

    def forward(self, rgb_in, depth_in, state_in, params=None):
        conv_out = self.forward_conv(rgb_in, depth_in, params)
        predict_pose = self.forward_predict_pose(conv_out, params)
        fc_out = self.forward_fc(conv_out, predict_pose, state_in, params)
        pi, sigma, mu, discrete = self.forward_action(fc_out, params)
        return pi, sigma, mu, discrete

    def sample_action(self, pi, sigma, mu, discrete, num_samples: int):
        batch_size = pi.shape[0]  # B
        targets = [
            mdn.sample(pi, sigma, mu) for _ in range(num_samples)
        ]  # S, B, O
        probs = [
            mdn.gaussian_probability(sigma, mu, t).log().sum(dim=1)
            for t in targets
        ]  # S, B
        probs = torch.cat([p.unsqueeze(1) for p in probs], dim=1)  # B, S
        target_maxs = torch.cat([
            # choose max O along S for each batch
            targets[prob_argmax][i].unsqueeze(0)  # 1, O
            for i, prob_argmax in enumerate(probs.argmax(dim=1))  # B
        ], dim=0)  # B, O
        return torch.cat((target_maxs, discrete), dim=1)  # B, O+1


def main(argv):
    model = Daml()
    device = model.device
    print(model)

    batch_size = 4
    im_width = FLAGS.im_width
    im_height = FLAGS.im_height
    num_channels = FLAGS.num_channels
    state_size = FLAGS.state_size
    action_size = FLAGS.action_size
    mdn_samples = FLAGS.mdn_samples

    rgb_in = torch.rand(
        [batch_size, num_channels, im_height, im_width], device=device)
    depth_in = torch.rand([batch_size, 1, im_height, im_width], device=device)
    state_in = torch.rand([batch_size, state_size], device=device)
    action_gt = torch.rand([batch_size, action_size], device=device)

    params = OrderedDict(model.meta_named_parameters())
    for name, param in params.items():
        print(name, param.shape)

    print("--------")
    for name, param in model.get_subdict(params, 'convs').items():
        print(name, param.shape)

    print("---------")
    print(params['rgb_bt'].shape)

    with Timer('forward once'):
        pi, sigma, mu, discrete = model.forward(
            rgb_in, depth_in, state_in, params)
        action = model.sample_action(pi, sigma, mu, discrete, mdn_samples)
        print(action.shape)


if __name__ == '__main__':
    app.run(main)
