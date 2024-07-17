from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import logging
from scipy import ndimage
import numpy as np
from networks import vit_seg_configs as configs
from networks.vit_seg_modeling_resnet_skip import ResNetV2
import torch.nn.functional as F
from networks.LeViT import LeViT_UNet_128s
# from networks.GAM_attention import simam_module
# from networks.giraffe_fpn_btn import CSPSMerge, GiraffeNeckV2
from networks.Attention import SpatialAttention_small, SpatialAttention, se_module, SCSEModule


logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,  
            out_channels,  
            skip_channels=0,  
            img_size=256,
            use_batchnorm=True,
    ):
        super().__init__()


        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.skip = skip_channels
        if out_channels == 256:
            self.att = se_module(out_channels)
        if out_channels == 128:
            self.att = SCSEModule(out_channels)
        if out_channels == 64:
            self.att = SpatialAttention(img_size=img_size)

    def forward(self, x, skip=None):

        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.skip != 0:
            y = self.att(x)
            return x + y
        else:
            return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        head_channels = 512

        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:  # 3
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, img_size = img_size) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):

        x = self.conv_more(hidden_states)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            # CNN 网络部分
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        # Embedded Sequence
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        x = x.flatten(2)

        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        # print(embeddings.shape)
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)

        return embedding_output, features


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class Attention_Guided_Adaptive_Fusion(nn.Module):
    def __init__(self, rfb=True):
        super(Attention_Guided_Adaptive_Fusion, self).__init__()

        compress_c = 16  # 8, 16, 32

        self.compress_level_0 = add_conv(1024, 768, 1, 1)
        self.se = se_module(768)

        self.expand = add_conv(768, 768, 3, 1)

        self.weight_level_0 = add_conv(768, compress_c, 1, 1)
        self.weight_level_1 = add_conv(768, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x_level_0, x_level_1):

        level_0_resized = self.compress_level_0(x_level_0)
        level_1_resized = x_level_1 + self.se(x_level_1)

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # fused_out_reduced = (level_0_resized + level_0_resized * levels_weight[:,0:1,:,:]) + \
        #                     (level_1_resized + level_1_resized * levels_weight[:,1:2,:,:])

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :]

        out = self.expand(fused_out_reduced)

        return out


class FATCNet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(FATCNet, self).__init__()

        # encoder块
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        # self.resnet = Transformer(config, img_size, vis)
        self.resnet = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        self.LeViT = LeViT_UNet_128s(img_size)
        self.fuse = Attention_Guided_Adaptive_Fusion(rfb=True)

        self.decoder = DecoderCup(config, img_size)

        # 分割
        # print(config['n_classes'])
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=2,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        trans_data2 = self.LeViT(x)

        x, features = self.resnet(x)

        x = self.fuse(x, trans_data2)
        
        x = self.decoder(x, features)

        logits = self.segmentation_head(x)

        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            # END
            self.resnet.root.conv.weight.copy_(
                np2th(res_weight["conv_root/kernel"], conv=True))
            gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            self.resnet.root.gn.weight.copy_(gn_weight)
            self.resnet.root.gn.bias.copy_(gn_bias)

            for bname, block in self.resnet.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'R50-ViT-B_16': configs.get_r50_b16_config(),
}
