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
    cfg.MODEL.OneNet.NMS = False  # for ablation
    cfg.MODEL.OneNet.HEAD = "CenterNet" # or FCOS or RetinaNet
    cfg.MODEL.OneNet.PRE_DEFINE = False
    
    # CenterNet Head.
    cfg.MODEL.OneNet.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.OneNet.ACTIVATION = 'relu'
    cfg.MODEL.OneNet.DECONV_CHANNEL= [2048, 256, 128, 64]
    cfg.MODEL.OneNet.DECONV_KERNEL = [4, 4, 4]
    cfg.MODEL.OneNet.DCN = True
    cfg.MODEL.OneNet.MODULATE_DEFORM = True
    
    # FCOS or RetinaNet Head.
    # cfg.MODEL.OneNet.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.OneNet.FEATURES_STRIDE = [8, 16, 32, 64, 128]
    cfg.MODEL.OneNet.OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],]
    cfg.MODEL.OneNet.NUM_CONV = 4
    cfg.MODEL.OneNet.CONV_NORM = "GN"
    cfg.MODEL.OneNet.CONV_CHANNELS = 256    
    
    # Cost.
    cfg.MODEL.OneNet.CLASS_COST = 2.0
    cfg.MODEL.OneNet.GIOU_COST = 2.0
    cfg.MODEL.OneNet.L1_COST = 5.0
    
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
