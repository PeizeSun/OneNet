#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec,batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, MinCostMatcher
from .head import Head
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

__all__ = ["OneNet"]


@META_ARCH_REGISTRY.register()
class OneNet(nn.Module):
    """
    Implement OneNet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        
        self.nms = cfg.MODEL.OneNet.NMS
        self.in_features = cfg.MODEL.OneNet.IN_FEATURES
        self.num_classes = cfg.MODEL.OneNet.NUM_CLASSES
        self.num_boxes = cfg.TEST.DETECTIONS_PER_IMAGE

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Head.
        self.head = Head(cfg=cfg, backbone_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.OneNet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.OneNet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.OneNet.L1_WEIGHT

        # Build Criterion.
        matcher = MinCostMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      losses=losses)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Cls & Reg Prediction.
        outputs_class, outputs_coord = self.head(features)
        
        output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, _box_cls, _box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape   (batch_size, K, H, W).
            box_pred (Tensor): tensors of shape (batch_size, 4, H, W).
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        box_cls = _box_cls.flatten(2)
        box_pred = _box_pred.flatten(2)
        
        assert len(box_cls) == len(image_sizes)
        results = []
        
        scores = torch.sigmoid(box_cls)

        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, box_pred, image_sizes
        )):
            result = Instances(image_size)
            
            # refer to https://github.com/FateScript/CenterNet-better
            topk_score_cat, topk_inds_cat = torch.topk(scores_per_image, k=self.num_boxes)
            topk_score, topk_inds = torch.topk(topk_score_cat.reshape(-1), k=self.num_boxes)
            topk_clses = topk_inds // self.num_boxes
            scores_per_image = topk_score
            labels_per_image = topk_clses
            
            topk_box_cat = box_pred_per_image[:, topk_inds_cat.reshape(-1)]
            topk_box = topk_box_cat[:, topk_inds]
            box_pred_per_image = topk_box.transpose(0, 1)
            
            if self.nms:
                keep = batched_nms(box_pred_per_image, 
                                   scores_per_image, 
                                   labels_per_image, 
                                   0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]

#         images = ImageList.from_tensors(images, self.size_divisibility)
        images = ImageList.from_tensors(images, 32)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
