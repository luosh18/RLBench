from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaModule


class MetaGroupNorm(nn.GroupNorm, MetaModule):
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.group_norm(
            input, self.num_groups, params['weight'], bias, self.eps)
