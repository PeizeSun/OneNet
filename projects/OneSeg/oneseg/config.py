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
    cfg.MODEL.OneNet.HEAD = "FCOS"


    # Head.
#     cfg.MODEL.OneNet.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.OneNet.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.OneNet.FEATURES_STRIDE = [8, 16, 32, 64, 128]
    cfg.MODEL.OneNet.NUM_CONV = 4
    cfg.MODEL.OneNet.CONV_NORM = "GN"
    cfg.MODEL.OneNet.CONV_CHANNELS = 256

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

    cfg.MODEL.CONDINST = CN()
    cfg.MODEL.CONDINST.SIZES_OF_INTEREST = [64, 128, 256, 512]
    # the downsampling ratio of the final instance masks to the input image
    cfg.MODEL.CONDINST.MASK_OUT_STRIDE = 4
    cfg.MODEL.CONDINST.BOTTOM_PIXELS_REMOVED = -1

    # if not -1, we only compute the mask loss for MAX_PROPOSALS random proposals PER GPU
    cfg.MODEL.CONDINST.MAX_PROPOSALS = -1
    # if not -1, we only compute the mask loss for top `TOPK_PROPOSALS_PER_IM` proposals
    # PER IMAGE in terms of their detection scores
    cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = -1

    cfg.MODEL.CONDINST.MASK_HEAD = CN()
    cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS = 16
    cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
    cfg.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
    cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

    cfg.MODEL.CONDINST.MASK_BRANCH = CN()
    cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
    cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
    cfg.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
    cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
    cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False
