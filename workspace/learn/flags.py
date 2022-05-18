import logging
from absl import app, flags

FLAGS = flags.FLAGS

# Dataset/Method options
flags.DEFINE_integer('im_width', 128,
                     'width of the images in the demo videos -- 125 for sim_push, 128 to suit RLBench')
flags.DEFINE_integer('im_height', 128,
                     'height of the images in the demo videos -- 125 for sim_push, 128 to suit RLBench')
flags.DEFINE_integer('num_channels', 3,
                     'number of channels of the images in the demo videos')
flags.DEFINE_integer('state_size', 9,
                     'dimension of robot state -- 9 for Jaco (exclude gripper, include tip position)')
flags.DEFINE_integer('action_size', 7,
                     'dimension of robot action -- 7 for Jaco (include gripper)')
flags.DEFINE_integer('T', 50,
                     'time horizon of the demo videos -- 50 for reach, 100 for push, DAML to be determined')
flags.DEFINE_float('simulation_timestep', 0.1,  # TODO: remember to set simulation_timestep while testing
                   'default 0.1 second for each frame')
flags.DEFINE_float('adapt_lr', '0.005',
                   'step size alpha for inner gradient update -- 0.005 for p&p')
flags.DEFINE_string('dataset_root', '/home/cscg-east92007/disk/dataset',
                    'roor directory of saved dataset')
flags.DEFINE_string('task_name', 'pick_and_place',
                    'task name (rlbench) not need to change?')
flags.DEFINE_integer('test_time', 15,
                     'time duration limit during testing (in second)')
flags.DEFINE_bool('gripper_action', False,
                  'use fixed gripper action for training, do not change state_size, this well auto-increase it')


# Training Options
flags.DEFINE_integer('dataset_seed', 42,
                     'random seed for dataset')
flags.DEFINE_integer('iteration', 75000,
                     'number of meta-training iterations  -- 75000 for p&p')
flags.DEFINE_integer('save_iter', 500,
                     'iteration interval for model saving')
flags.DEFINE_integer('log_iter', 100,
                     'iteration interval for loggin loss')
flags.DEFINE_string('save_dir', '/home/cscg-east92007/disk/save',
                    'dir to save models')
flags.DEFINE_bool('cuda', True, 'using GPU, default true')
flags.DEFINE_integer('meta_batch_size', 4,
                     'number of tasks sampled per meta-update -- 4 for p&p')
flags.DEFINE_integer('num_updates', 5,
                     'number of inner gradient updates during training -- 5 for p&p')
flags.DEFINE_float('inner_clip', 30.0,
                   'inner gradient clipping  -- [-30, 30] for p&p')

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
flags.DEFINE_integer('predict_size', 3,
                     'dimension of end-effector position prediction (tip x,y,z)')
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


def test(argv):
    f = FLAGS.flag_values_dict()
    # print(f['temp_filter_size'])
    logging.info('FLAGS: %s' % str(f))


if __name__ == '__main__':
    app.run(test)
