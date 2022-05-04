import torch
import torch.nn as nn


def init_weights_xavier(weights):
    # init weights with xavier uniform initi
    return nn.init.xavier_uniform_(weights)


def init_weights_zeros(weights):
    # init weights with zeros value
    return nn.init.zeros_(weights)


def init_weights_normal(weights, mean=0.0, std=1.0):
    # init weights with normal distribution
    return nn.init.normal_(weights, mean=mean, std=std)


def zero_padding(shape, conv1d=False):
    if(conv1d):
        return nn.ConstantPad1d(shape, 0)
    else:
        return nn.ZeroPad2d(shape)
