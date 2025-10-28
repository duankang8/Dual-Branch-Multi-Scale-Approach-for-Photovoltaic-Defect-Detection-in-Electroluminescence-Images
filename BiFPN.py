import torch
import torch.nn as nn


class Concat_BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(Concat_BiFPN, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)

"""
#新的版本
import torch.nn as nn
import torch
 
class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
 
class Bi_FPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001
 
    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon) # 权重归一化处理
        weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result
        
from .modules.bifpn import Bi_FPN      
#tasks.py中parse_model的改动替换
elif m in {bifpn}:
length = len([ch[x] for x in f])
args = [length]



import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        return self.relu(x)


class BiFPN(nn.Module):
    def __init__(self, num_channels, num_levels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.num_levels = num_levels

        # 可学习的权重初始化
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(num_levels - 1)])
        self.up_convs = nn.ModuleList([SeparableConvBlock(num_channels, num_channels) for _ in range(num_levels - 1)])
        self.down_convs = nn.ModuleList([SeparableConvBlock(num_channels, num_channels) for _ in range(num_levels - 1)])

    def forward(self, inputs):
        # 上采样路径
        up_features = [inputs[-1]]  # 从最顶层开始
        for i in range(self.num_levels - 1):
            w = F.relu(self.weights[i])
            w = w / (torch.sum(w) + self.epsilon)  # 归一化权重
            up_feature = w[0] * inputs[-(i + 2)] + w[1] * F.interpolate(up_features[-1], scale_factor=2, mode='nearest')
            up_feature = self.up_convs[i](up_feature)
            up_features.append(up_feature)

        # 下采样路径
        down_features = [up_features[-1]]  # 从最顶层开始
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.weights[i])
            w = w / (torch.sum(w) + self.epsilon)  # 归一化权重
            down_feature = w[0] * up_features[i] + w[1] * F.max_pool2d(down_features[-1], 2)
            down_feature = self.down_convs[i](down_feature)
            down_features.append(down_feature)

        down_features.reverse()
        return down_features
"""

