import os
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import workspace.models.mdn as mdn
from absl import app
from torchmeta.modules import (MetaConv1d, MetaConv2d, MetaLinear, MetaModule,
                               MetaSequential)
from workspace.learn.flags import FLAGS
from workspace.models.meta_groupnorm import MetaGroupNorm
from workspace.models.saptial_softmax import SpatialSoftmax
from workspace.models.vector_norm import VectorNorm
from workspace.utils import Timer, check_and_make

""" note for input, output, target
rgb:        (3, 128, 128)
depth:      (1, 128, 128)
state:      (10)
action:     (7)
predict:    (3)
"""

MODEL_FOLDER = 'model'


class Daml(MetaModule):
    def __init__(self) -> None:
        super().__init__()
        self.state_size = FLAGS.state_size
        self.action_size = FLAGS.action_size
        self.inner_clip = FLAGS.inner_clip
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
        self.predict_pose = MetaLinear(
            self.num_feature_output, FLAGS.predict_size)

        ###### fully-connected layers ######
        # input: cat: feature (spatial softmax), predicted_pose, robot state, bias transform
        fc_shape_in = self.num_feature_output + \
            FLAGS.predict_size + self.state_size + FLAGS.bt_dim
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
                    MetaGroupNorm(1, num_filter),
                    nn.ReLU(),
                ])
            temp_convs.extend([
                MetaConv1d(num_temp_filters[-1], 1, 1, padding='same'),
                VectorNorm(),  # perform L2 vector norm after convs (adaptive loss)
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
            elif isinstance(m, nn.ReLU) or isinstance(m, VectorNorm):
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
        for m in [*self.mdn.children(), *self.discrete.children()]:
            if isinstance(m, MetaLinear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # register parameters to update during adaption
        self.adapt_prefixes = [
            'rgb_bt', 'depth_bt', 'fc_bt',
            'rgb_conv', 'depth_conv', 'convs',
            'fcs',
        ]

    def forward_conv(self, rgb_in, depth_in, params) -> torch.Tensor:
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

    def forward_predict_pose(self, conv_out, params) -> torch.Tensor:
        # outer meta-objective
        # predict the pose of the gripper when it contacts
        # the target object and/or container by linear (V. A.)
        return self.predict_pose(conv_out, self.get_subdict(params, 'predict_pose'))

    def forward_fc(self, conv_out, predict_pose, state_in, params) -> torch.Tensor:
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

    def forward_action(self, fc_out, params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MDN cat with Linear gripper output"""
        pi, sigma, mu = self.mdn(
            fc_out, params=self.get_subdict(params, 'mdn'))
        discrete = self.discrete(
            fc_out, params=self.get_subdict(params, 'discrete'))
        return pi, sigma, mu, discrete

    def forward(self, rgb_in, depth_in, state_in, params):
        conv_out = self.forward_conv(rgb_in, depth_in, params)
        predict_pose = self.forward_predict_pose(conv_out, params)
        fc_out = self.forward_fc(conv_out, predict_pose, state_in, params)
        pi, sigma, mu, discrete = self.forward_action(fc_out, params)
        return pi, sigma, mu, discrete, predict_pose

    def sample_action(self, pi, sigma, mu, discrete, num_samples: int):
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

    def _adapt_loss(self, rgb: torch.Tensor, depth: torch.Tensor, batch_params: List[OrderedDict]) -> torch.Tensor:
        """loss for inner gradient update (video only) (private function)"""
        # get feature points & fc output
        fps = [self.forward_conv(rgb_in, depth_in, params)
               for rgb_in, depth_in, params in zip(rgb, depth, batch_params)]  # B[T, fp]
        fcs = [self.forward_fc(conv_out, self.forward_predict_pose(conv_out, params), None, params)
               for conv_out, params in zip(fps, batch_params)]  # B[T, fc]
        # reshape
        fps = [fp.t().unsqueeze(0) for fp in fps]  # B[1, fp, T]
        fcs = [fc.t().unsqueeze(0) for fc in fcs]  # B[1, fc, T]
        # temp conv (adaptive loss)
        loss = [
            self.feature_temp_convs(
                fp, params=self.get_subdict(params, 'feature_temp_convs')
            ) + self.fc_temp_convs(
                fc, params=self.get_subdict(params, 'fc_temp_convs')
            )
            for fp, fc, params in zip(fps, fcs, batch_params)
        ]  # B[1]
        return loss

    def adapt(self, rgb: torch.Tensor, depth: torch.Tensor, params: OrderedDict,
              step_size=0.05, updates=1) -> Tuple[List[OrderedDict], List[torch.Tensor]]:
        """perform adaptive gradient update "updates" times for each task in batch
        rgb: [B, T, C, H, W]; depth: [B, T, 1, H, W];
        T is the length of demo (sampled)
        """
        batch_params = [
            OrderedDict([
                (name,
                 p.clone() if name.split('.', 1)[0] in self.adapt_prefixes else p)  # reduce memory usage
                for name, p in params.items()
            ]) for _ in range(rgb.shape[0])
        ]  # clone param for each task in batch -> B[params]

        def update_params(loss, params):
            params_to_update = OrderedDict([
                (name, params[name])
                for name in params if name.split('.', 1)[0] in self.adapt_prefixes
            ])
            grads = torch.autograd.grad(loss, params_to_update.values(),
                                        create_graph=True)  # create_graph to allow higher order gradient
            for (name, param), grad in zip(params_to_update.items(), grads):
                params[name] = param - step_size * \
                    grad.clip(-self.inner_clip, self.inner_clip)
            return params

        adapt_losses = []
        for _ in range(updates):
            adapt_loss = self._adapt_loss(rgb, depth, batch_params)
            # mean adapt loss across batch
            adapt_losses.append(sum(adapt_loss) / len(adapt_loss))
            batch_params = [
                update_params(loss, params)
                for loss, params in zip(adapt_loss, batch_params)
            ]

        return batch_params, adapt_losses  # mean adapt loss across batch for each update

    def meta_loss(self, rgb: torch.Tensor, depth: torch.Tensor, state: torch.Tensor,
                  target: torch.Tensor, predict_target: torch.Tensor,
                  batch_params: List[OrderedDict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """loss for meta gradient update (across batch)
        rgb: [B, T, C, H, W]; depth: [B, T, 1, H, W]; state: [B, T, state_size];
        target: [B, T, action_size]; predict_target: [B, T, predict_size];
        """
        # prepare target for continuos [B, T, action_size - 1] & discrete actions [B, T, 1]
        continuous_tg, discrete_tg = target.split([self.action_size - 1, 1], 2)
        outs = [
            self.forward(rgb_in, depth_in, state_in, params)
            for rgb_in, depth_in, state_in, params in zip(rgb, depth, state, batch_params)
        ]  # B[Tuple(pi, sigma, mu, discrete, predict_pose)]
        loss = torch.stack([
            torch.stack([  # sum across time
                mdn.mdn_loss(pi, sigma, mu, ct),
                F.binary_cross_entropy(discrete, dt, reduction='sum'),
                F.mse_loss(predict, pt, reduction='sum'),
            ])
            for (pi, sigma, mu, discrete, predict), ct, dt, pt
            in zip(outs, continuous_tg, discrete_tg, predict_target)
        ])  # B, 3
        # mean across batch & sum across loss, mean loss for each kind of loss
        return loss.sum(1).mean(), loss.mean(0)


def save_model(model: Daml, optim: torch.optim.Optimizer, name):
    save_dir = os.path.join(FLAGS.save_dir, MODEL_FOLDER)
    check_and_make(save_dir)
    filename = os.path.join(save_dir, name + '.pt')
    torch.save(model.state_dict(), filename)
    # save meta_named_parameters (backup)
    torch.save(OrderedDict(model.meta_named_parameters()),
               filename + '.backup')
    # save optimizer
    opt_filename = os.path.join(
        save_dir, name + '_optim.pt')
    torch.save(optim.state_dict(), opt_filename)


def load_model(model: Daml, optim: torch.optim.Optimizer, name):
    save_dir = os.path.join(FLAGS.save_dir, MODEL_FOLDER)
    filename = os.path.join(save_dir, name + '.pt')
    model.load_state_dict(torch.load(filename))
    opt_filename = os.path.join(save_dir, name + '_optim.pt')
    optim.load_state_dict(torch.load(opt_filename))


def test_params(model: Daml):
    params = OrderedDict(model.meta_named_parameters())
    for name, param in params.items():
        print(name, param.shape)
    print("--------")
    for name, param in model.get_subdict(params, 'convs').items():
        print(name, param.shape)
    print("---------")
    print(params['rgb_bt'].shape)


def test_forward(model):
    device = model.device
    batch_size = FLAGS.meta_batch_size
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

    params = OrderedDict(model.meta_named_parameters())

    with Timer('forward once'):
        pi, sigma, mu, discrete = model.forward(
            rgb_in, depth_in, state_in, params)
        action = model.sample_action(pi, sigma, mu, discrete, mdn_samples)
        print(action.shape)


def test_adapt_meta(model: Daml):
    device = model.device
    batch_size = FLAGS.meta_batch_size
    im_width = FLAGS.im_width
    im_height = FLAGS.im_height
    num_channels = FLAGS.num_channels
    state_size = FLAGS.state_size
    action_size = FLAGS.action_size
    predict_size = FLAGS.predict_size
    T = FLAGS.T

    rgb = torch.rand([batch_size, T, num_channels, im_height, im_width],
                     device=device)
    depth = torch.rand([batch_size, T, 1, im_height, im_width], device=device)
    state = torch.rand([batch_size, T, state_size], device=device)
    target = torch.rand([batch_size, T, action_size], device=device)
    predict_target = torch.rand([batch_size, T, predict_size], device=device)

    pre_update = OrderedDict(model.meta_named_parameters())
    meta_optimizer = torch.optim.Adam(model.parameters())
    model.zero_grad()
    for name, p in model.meta_named_parameters():
        print(name)

    # adapt (get post-update params)
    print('----- adapt -----')
    batch_post_update, adapt_losses = model.adapt(
        rgb, depth, pre_update, FLAGS.adapt_lr, FLAGS.num_updates)
    print('batch_post_update: len=', len(batch_post_update))
    print('adapt_losses:', len(adapt_losses), adapt_losses)
    for name in pre_update:
        for post_update in batch_post_update:
            if not torch.equal(pre_update[name], post_update[name]):
                print(name, torch.linalg.norm(
                    post_update[name] - pre_update[name]))

    # meta
    print('----- start meta -----')
    meta_loss, meta_loss_mat = model.meta_loss(rgb, depth, state,
                                               target, predict_target,
                                               batch_post_update)
    print('outer_loss:', meta_loss)
    print('meta_loss_mat', meta_loss_mat)
    meta_loss.backward()
    for name, p in model.named_parameters():
        if ('temp' in name):
            print(name, p.flatten()[0], p.grad.norm())

    meta_optimizer.step()
    for name, p in model.named_parameters():
        if ('temp' in name):
            print(name, p.flatten()[0], p.grad.norm())

    print('----- time for iterration -----')
    with Timer('10 Iterration'):
        for _ in range(10):
            model.zero_grad()
            pre_update = OrderedDict(model.meta_named_parameters())
            meta_optimizer = torch.optim.Adam(model.parameters())
            batch_post_update, _ = model.adapt(
                rgb, depth, pre_update, FLAGS.adapt_lr, FLAGS.num_updates)
            meta_loss, _ = model.meta_loss(rgb, depth, state,
                                           target, predict_target,
                                           batch_post_update)
            meta_loss.backward()
            meta_optimizer.step()


def test_save_load(model: Daml, optim: torch.optim.Optimizer):
    print('----- test_save_load -----')
    model_dict = model.state_dict()
    optim_dict = optim.state_dict()
    save_model(model, optim, 'test')
    load_model(model, optim, 'test')
    for k in model_dict:
        if not torch.equal(model_dict[k], model.get_parameter(k)):
            print(k)
    if not str(optim_dict) == str(optim.state_dict()):
        print(False)
    # check backup
    save_dir = FLAGS.save_dir
    filename = os.path.join(
        save_dir, MODEL_FOLDER, 'test.pt.backup')
    model.load_state_dict(torch.load(filename))
    for k in model_dict:
        if not torch.equal(model_dict[k], model.get_parameter(k)):
            print(k)


def main(argv):
    model = Daml()
    print(model)
    # test_params(model)
    # test_adapt_meta(model)
    # test_save_load(model, torch.optim.Adam(model.parameters()))


if __name__ == '__main__':
    app.run(main)
