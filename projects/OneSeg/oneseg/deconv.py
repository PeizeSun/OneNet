#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) https://github.com/FateScript/CenterNet-better
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#

import math
import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, DeformConv, ModulatedDeformConv

class DCNDeconvLayer(nn.Module):

    def __init__(
        self, in_planes,
        out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, modulate_deform=True,
        num_groups=1, deform_num_groups=1,
        dilation=1
    ):
        super(DCNDeconvLayer, self).__init__()
        self.deform_modulated = modulate_deform
        if modulate_deform:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18
        
        self.dcn_offset = Conv2d(
            in_planes,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            dilation=dilation
        )
        self.dcn = deform_conv_op(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups
        )
        for layer in [self.dcn]:
            weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)
        
        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        if self.deform_modulated:
            offset_mask = self.dcn_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.dcn(out, offset, mask)
        else:
            offset = self.dcn_offset(out)
            out = self.dcn(out, offset)
        x = out
        
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

class CNDeconvLayer(nn.Module):

    def __init__(
        self, 
        in_planes, out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1, deconv_out_pad=0,
        num_groups=1, dilation=1,
        modulate_deform=True # not used
    ):
        super(CNDeconvLayer, self).__init__()
        
        self.conv = Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=dilation,
        )

        for layer in [self.conv]:
            weight_init.c2_msra_fill(layer)
        
        self.bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

class CenternetDeconv(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg, backbone_shape):
        super(CenternetDeconv, self).__init__()
        channels = cfg.MODEL.OneNet.DECONV_CHANNEL
        deconv_kernel = cfg.MODEL.OneNet.DECONV_KERNEL
        modulate_deform = cfg.MODEL.OneNet.MODULATE_DEFORM
        in_features = cfg.MODEL.OneNet.IN_FEATURES        
        
        if cfg.MODEL.OneNet.DCN:
            DeconvLayer = DCNDeconvLayer
        else:
            DeconvLayer = CNDeconvLayer
        
        self.deconv1 = DeconvLayer(
            channels[0], channels[1],
            deconv_kernel=deconv_kernel[0],
            modulate_deform=modulate_deform,
        )

        self.lateral_conv1 = nn.Conv2d(
            backbone_shape[in_features[-2]].channels, 
            channels[1], 
            kernel_size=1, stride=1, padding=0)
        
        self.deconv2 = DeconvLayer(
            channels[1], channels[2],
            deconv_kernel=deconv_kernel[1],
            modulate_deform=modulate_deform,
        )
        self.lateral_conv2 = nn.Conv2d(
           backbone_shape[in_features[-3]].channels, 
            channels[2], 
            kernel_size=1, stride=1, padding=0)
        
        self.deconv3 = DeconvLayer(
            channels[2], channels[3],
            deconv_kernel=deconv_kernel[2],
            modulate_deform=modulate_deform,
        )
        self.lateral_conv3 = nn.Conv2d(
            backbone_shape[in_features[-4]].channels, 
            channels[3], 
            kernel_size=1, stride=1, padding=0)
        
        self.output_conv = nn.Conv2d(
            channels[3], channels[3], 
            kernel_size=3, stride=1, padding=1)
        
    def forward(self, features_list):
    
        x = features_list[-1]
        x = self.deconv1(x)
        x = self.lateral_conv1(features_list[-2]) + x
        x = self.deconv2(x)
        x = self.lateral_conv2(features_list[-3]) + x
        x = self.deconv3(x)
        x = self.lateral_conv3(features_list[-4]) + x
        x = self.output_conv(x)
        return x
