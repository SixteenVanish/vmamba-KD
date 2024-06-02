from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        # 每个元素平方，沿着channel计算均值，铺平成[bsz, H*W]
        # 默认情况下，F.normalize 对最后一个维度进行归一化，即对每个样本的向量 [H * W] 进行归一化
        # 即计算每个样本的L2范数，然后将每个元素除以这个范数，这使得归一化后的每个样本的 L2 范数为 1
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))   
    
class Attention_v2(nn.Module):
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        # 每个元素平方，沿着channel计算均值，铺平成[bsz, H*W]
        # 默认情况下，F.normalize 对最后一个维度进行归一化，即对每个样本的向量 [H * W] 进行归一化
        # 即计算每个样本的L2范数，然后将每个元素除以这个范数，这使得归一化后的每个样本的 L2 范数为 1
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))   