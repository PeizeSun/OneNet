#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OneNet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from .deconv import CenternetDeconv


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

    
class Head(nn.Module):

    def __init__(self, cfg, backbone_shape=[2048, 1024, 512, 256]):
        super().__init__()
        
        # Build heads.
        num_classes = cfg.MODEL.OneNet.NUM_CLASSES
        d_model = cfg.MODEL.OneNet.DECONV_CHANNEL[-1]
        activation = cfg.MODEL.OneNet.ACTIVATION

        self.deconv = CenternetDeconv(cfg, backbone_shape)
        
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_classes = num_classes
        self.activation = _get_activation_fn(activation)

        self.feat1 = nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1)
        self.cls_score = nn.Conv2d(d_model, num_classes, kernel_size=3, stride=1, padding=1)
        self.ltrb_pred = nn.Conv2d(d_model, 4, kernel_size=3, stride=1, padding=1)        
        
        # Init parameters.
        prior_prob = cfg.MODEL.OneNet.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_score.bias, self.bias_value)
    
    def forward(self, features_list):
        
        features = self.deconv(features_list)
        locations = self.locations(features)[None]       

        feat = self.activation(self.feat1(features))
    
        class_logits = self.cls_score(feat)
        pred_ltrb = F.relu(self.ltrb_pred(feat))
        pred_bboxes = self.apply_ltrb(locations, pred_ltrb)

        return class_logits, pred_bboxes
    
    def apply_ltrb(self, locations, pred_ltrb): 
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W) 
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:,0,:,:] = locations[:,0,:,:] - pred_ltrb[:,0,:,:]  # x1
        pred_boxes[:,1,:,:] = locations[:,1,:,:] - pred_ltrb[:,1,:,:]  # y1
        pred_boxes[:,2,:,:] = locations[:,0,:,:] + pred_ltrb[:,2,:,:]  # x2
        pred_boxes[:,3,:,:] = locations[:,1,:,:] + pred_ltrb[:,3,:,:]  # y2

        return pred_boxes    
    
    @torch.no_grad()
    def locations(self, features, stride=4):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2            
        
        locations = locations.reshape(h, w, 2).permute(2, 0, 1)
        
        return locations



class FCOSHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Build heads.
        num_classes = cfg.MODEL.OneNet.NUM_CLASSES
        d_model = cfg.MODEL.OneNet.CONV_CHANNELS
        activation = cfg.MODEL.OneNet.ACTIVATION
        num_conv = cfg.MODEL.OneNet.NUM_CONV
        conv_norm = cfg.MODEL.OneNet.CONV_NORM
        num_levels = len(cfg.MODEL.OneNet.IN_FEATURES)
        conv_channels = cfg.MODEL.OneNet.CONV_CHANNELS

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_levels)])
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_classes = num_classes
        self.activation = _get_activation_fn(activation)
        self.features_stride = cfg.MODEL.OneNet.FEATURES_STRIDE
#         self.dropblock = cfg.MODEL.OneNet.DROPBLOCK
#         self.dropblock_size = cfg.MODEL.OneNet.DROPBLOCK_SIZE
#         self.dropblock_prob = cfg.MODEL.OneNet.DROPBLOCK_PROB
#         self.coord_conv = cfg.MODEL.OneNet.COORD_CONV
        self.coord_conv = False
        
        cls_conv_module = list()
        for idx in range(num_conv):
#             if self.dropblock:
#                 cls_conv_module.append(DropBlock(block_size=self.dropblock_size, drop_prob=self.dropblock_prob))
            if idx == 0:
                cls_conv_module.append(nn.Conv2d(d_model, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
            else:
                cls_conv_module.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
            if conv_norm == "GN":
                cls_conv_module.append(nn.GroupNorm(32, conv_channels))
            else:
                cls_conv_module.append(nn.BatchNorm2d(conv_channels))
            cls_conv_module.append(nn.ReLU(inplace=True))

        self.cls_conv_module = nn.ModuleList(cls_conv_module)

        reg_conv_module = list()
        for idx in range(num_conv):
#             if self.dropblock:
#                 reg_conv_module.append(DropBlock(block_size=self.dropblock_size, drop_prob=self.dropblock_prob))
            if idx == 0:
                if self.coord_conv:
                    reg_conv_module.append(nn.Conv2d(d_model + 2, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
                else:
                    reg_conv_module.append(nn.Conv2d(d_model, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
            else:
                if self.coord_conv:
                    reg_conv_module.append(nn.Conv2d(conv_channels + 2, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
                else:
                    reg_conv_module.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
            if conv_norm == "GN":
                reg_conv_module.append(nn.GroupNorm(32, conv_channels))
            else:
                reg_conv_module.append(nn.BatchNorm2d(conv_channels))
            reg_conv_module.append(nn.ReLU(inplace=True))

        self.reg_conv_module = nn.ModuleList(reg_conv_module)

        # self.feat1 = nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1)
        self.cls_score = nn.Conv2d(conv_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.ltrb_pred = nn.Conv2d(conv_channels, 4, kernel_size=3, stride=1, padding=1)

        # Init parameters.
        prior_prob = cfg.MODEL.OneNet.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)

    def forward(self, features_list, top_module=None):

        # features = self.deconv(features_list)
        class_logits = list()
        pred_bboxes = list()
        top_feats = list()
        locationall = list()
        fpn_levels = list()
        
        for l, stride_feat in enumerate(features_list):
            cls_feat = stride_feat
            reg_feat = stride_feat

            if self.coord_conv:
                x_range = torch.linspace(-1, 1, reg_feat.shape[-1], device=reg_feat.device)
                y_range = torch.linspace(-1, 1, reg_feat.shape[-2], device=reg_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([reg_feat.shape[0], 1, -1, -1])
                x = x.expand([reg_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                # reg_feat = torch.cat([reg_feat, coord_feat], 1)
                # cls_feat = torch.cat([cls_feat, coord_feat], 1)

            for conv_layer in self.cls_conv_module:
                cls_feat = conv_layer(cls_feat)

            for conv_layer in self.reg_conv_module:
                if self.coord_conv and isinstance(conv_layer, nn.Conv2d):
                    reg_feat = torch.cat([reg_feat, coord_feat], 1)
                reg_feat = conv_layer(reg_feat)

            locations = self.locations(stride_feat, self.features_stride[l])[None]
                
            stride_class_logits = self.cls_score(cls_feat)    
            reg_ltrb = self.ltrb_pred(reg_feat)

            scale_reg_ltrb = self.scales[l](reg_ltrb)
            stride_pred_ltrb = F.relu(scale_reg_ltrb) * self.features_stride[l]
            stride_pred_bboxes = self.apply_ltrb(locations, stride_pred_ltrb)
            bs, c, h, w = stride_class_logits.shape
            bs, k, h, w = stride_pred_bboxes.shape
            class_logits.append(stride_class_logits.view(bs, c, -1))
            pred_bboxes.append(stride_pred_bboxes.view(bs, k, -1))
            

            locationall.append(locations.view(1, 2, -1).repeat(bs,1,1))            
            fpn_levels.append(locations.new_ones(bs, 1, h*w)*l)
            
            if top_module is not None:
                top_feat = top_module(reg_feat)
                bs, c, h, w = top_feat.shape
                top_feats.append(top_feat.view(bs, c, -1))
                
        class_logits = torch.cat(class_logits, dim=-1)
        pred_bboxes = torch.cat(pred_bboxes, dim=-1)
        locationall = torch.cat(locationall, dim=-1)
        fpn_levels = torch.cat(fpn_levels, dim=-1)
        
        if len(top_feats) > 0:
            top_feats = torch.cat(top_feats, dim=-1)
        else:
            top_feats = None
        
        return class_logits, pred_bboxes, top_feats, locationall, fpn_levels

    def apply_ltrb(self, locations, pred_ltrb):
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W)
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:, 0, :, :] = locations[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        pred_boxes[:, 1, :, :] = locations[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        pred_boxes[:, 2, :, :] = locations[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        pred_boxes[:, 3, :, :] = locations[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2

        return pred_boxes

    @torch.no_grad()
    def locations(self, features, stride=4):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device

        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.reshape(h, w, 2).permute(2, 0, 1)

        return locations


    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
