# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License

import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
# from networks.AxialAttention import RowAttention,ColAttention


specification = {
    'LeViT_128S': {
        'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'},
}

__all__ = [specification.keys()]



FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (resolution**4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**4)
        #attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution**4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**2) * (resolution_**2)
        #attention * v
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT_UNet_128s(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=640):
        super().__init__()
        global FLOPS_COUNTER

        num_heads = [4, 6, 8]
        depth = [2, 3, 4]
        patch_size = 16

        key_dim = [16, 16, 16]
        attn_ratio = [2, 2, 2]
        mlp_ratio = [2, 2, 2]
        down_ops = [
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 16, 128 // 16, 4, 2, 2],
            ['Subsample', 16, 256 // 16, 4, 2, 2],
        ]
        img_size = img_size
        act = torch.nn.Hardswish
        attention_activation = act
        mlp_activation = act
        drop_path = 0
        distillation = True

        self.num_classes = 2
        self.num_features = 384
        self.embed_dim = embed_dim = [128, 256, 384]
        self.distillation = distillation

        # self.patch_embed = hybrid_backbone ## aping: CNN  # aping num_channel
        n = 128 ## cnn num channel
        activation = torch.nn.Hardswish
        self.cnn_b1 = torch.nn.Sequential(
            Conv2d_BN(3, n // 8, 3, 2, 1, resolution=img_size), activation())
        self.cnn_b2 = torch.nn.Sequential(
            Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=img_size // 2), activation())
        self.cnn_b3 = torch.nn.Sequential(
            Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=img_size // 4), activation())
        self.cnn_b4 = torch.nn.Sequential(
            Conv2d_BN(n // 2, n, 3, 2, 1, resolution=img_size // 8))

        self.blocks = [] ## attention block
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            print(i)
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
        self.blocks = torch.nn.Sequential(*self.blocks)
        ## divid the blocks
        self.block_1 = self.blocks[0:4]
        self.block_2 = self.blocks[4:12]
        self.block_3 = self.blocks[12:]

        del self.blocks

        ## aping: upsampling
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x_cnn_1 = self.cnn_b1(x)
        x_cnn_2 = self.cnn_b2(x_cnn_1)
        x_cnn_3 = self.cnn_b3(x_cnn_2)
        x_cnn = self.cnn_b4(x_cnn_3)

        x = x_cnn.flatten(2).transpose(1, 2)

        ## aping 
        x = self.block_1(x)  # torch.Size([1, 196, 128])--> torch.Size([1, 196, 128]) : CNN->Trans
        x_num, x_len = x.shape[0], x.shape[1]
        # reshape: torch.Size([4, 14, 14, 256])
        x_r_1 = x.reshape(x_num, int(x_len**0.5), int(x_len**0.5), -1)
        x_r_1 = x_r_1.permute(0, 3, 1, 2)

        x = self.block_2(x) # downsample + att torch.Size([1, 196, 128])-> torch.Size([1, 49, 256])
        x_num, x_len = x.shape[0], x.shape[1]
        # reshape: torch.Size([4, 14, 14, 256])
        x_r_2 = x.reshape(x_num, int(x_len**0.5), int(x_len**0.5), -1)
        x_r_2 = x_r_2.permute(0, 3, 1, 2)
        ## upsampling
        x_r_2_up = self.up(x_r_2)
        

        x = self.block_3(x) # torch.Size([1, 49, 256])->torch.Size([1, 16, 384])
        x_num, x_len = x.shape[0], x.shape[1]
        # reshape: torch.Size([4, 4, 4, 512])
        x_r_3 = x.reshape(x_num, int(x_len**0.5), int(x_len**0.5), -1)
        x_r_3 = x_r_3.permute(0, 3, 1, 2)
        ## upsampling
        x_r_3_up = self.up(x_r_3)
        x_r_3_up = self.up(x_r_3_up)
        
        ## aping: resize the feature maps
        if (x_r_2_up.shape  != x_r_3_up.shape):
            x_r_3_up = F.interpolate(x_r_3_up, size=x_r_2_up.shape[2:], mode="bilinear",  align_corners=True)
        att_all = torch.cat([x_r_1, x_r_2_up, x_r_3_up], dim=1)  # 128+256+384=768

        # att_all = att_all.flatten(2).transpose(-1, -2)

        return att_all

