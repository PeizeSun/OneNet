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
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)
    
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


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
