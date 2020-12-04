# -*- coding: utf-8 -*-
#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_onenet_config(cfg):
    """
    Add config for OneNet.
    """
    cfg.MODEL.OneNet = CN()
    cfg.MODEL.OneNet.NUM_CLASSES = 80

    # Head.
    cfg.MODEL.OneNet.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.OneNet.ACTIVATION = 'relu'
    cfg.MODEL.OneNet.NMS = False  # for ablation
    
    # Deconv
    cfg.MODEL.OneNet.DECONV_CHANNEL= [2048, 256, 128, 64]
    cfg.MODEL.OneNet.DECONV_KERNEL = [4, 4, 4]
    cfg.MODEL.OneNet.DCN = True
    cfg.MODEL.OneNet.MODULATE_DEFORM = True
    
    # Loss.
    cfg.MODEL.OneNet.CLASS_WEIGHT = 2.0
    cfg.MODEL.OneNet.GIOU_WEIGHT = 2.0
    cfg.MODEL.OneNet.L1_WEIGHT = 5.0
    
    # Focal Loss.
    cfg.MODEL.OneNet.ALPHA = 0.25
    cfg.MODEL.OneNet.GAMMA = 2.0
    cfg.MODEL.OneNet.PRIOR_PROB = 0.01
    
    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
