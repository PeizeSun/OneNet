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
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, MinCostMatcher
from .head import Head, FCOSHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .mask_branch import build_mask_branch
from .mask_head_dynamic import build_dynamic_mask_head
from .comm import aligned_bilinear

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
        self.head = FCOSHead(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        
        # build top module
        in_channels = self.backbone.output_shape()[self.in_features[0]].channels
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        
        # Loss parameters:
        class_weight = cfg.MODEL.OneNet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.OneNet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.OneNet.L1_WEIGHT
        mask_weight = 2
        
        # Build Criterion.
        matcher = MinCostMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, "loss_mask": mask_weight}

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
        outputs_class, outputs_coord, outputs_top_feat, locations, fpn_levels = self.head(features, self.controller)
        
        mask_feats, _ = self.mask_branch(src)
        
        output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord, 'pred_top_feats': outputs_top_feat,
                 'locations': locations, 'fpn_levels': fpn_levels}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict, indices, indices_aug = self.criterion(output, targets)
            
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
#             mask_losses = self._forward_mask_heads_train(output, mask_feats, indices, gt_instances)
            mask_losses = self._forward_mask_heads_train(output, mask_feats, indices_aug, gt_instances)

            loss_dict.update(mask_losses)
            
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            
            return loss_dict

        else:
            results = self.inference(outputs_class, outputs_coord, images.image_sizes, locations, fpn_levels, outputs_top_feat)
            
            pred_instances_w_masks = self._forward_mask_heads_test(results, mask_feats)
            
            padded_im_h, padded_im_w = images.tensor.size()[-2:]

            processed_results = []
            for im_id, (results_per_image, input_per_image, image_size) in enumerate(zip(results, batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                
                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]

#                 r = detector_postprocess(results_per_image, height, width)
                r = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )
                processed_results.append({"instances": r})
            
            return processed_results

    def _forward_mask_heads_train(self, output, mask_feats, indices, gt_instances):

        locations, fpn_levels, top_feat = output['locations'], output['fpn_levels'], output['pred_top_feats']
        
        batch_instances = []
        num_gts = 0        
        for im_id, ind_per_im in enumerate(indices):
            sample_ind_per_im, gt_ind_per_im = ind_per_im
            instances = Instances((0, 0))
            instances.locations = locations[im_id][:,sample_ind_per_im].transpose(0, 1)
            instances.fpn_levels = fpn_levels[im_id][:,sample_ind_per_im].transpose(0, 1)
            instances.top_feats = top_feat[im_id][:,sample_ind_per_im].transpose(0, 1)
            instances.gt_inds = gt_ind_per_im + num_gts
            instances.im_inds = instances.gt_inds.new_ones(len(gt_ind_per_im), dtype=torch.long) * im_id

            num_gts += len(gt_ind_per_im)//2
            
            batch_instances.append(instances)
            
        pred_instances = Instances.cat(batch_instances)
        pred_instances.mask_head_params = pred_instances.top_feats

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances
        )

        return loss_mask
    
    def _forward_mask_heads_test(self, proposals, mask_feats):

        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.scores.new_ones(len(per_im), dtype=torch.long) * im_id

        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances
        )

        return pred_instances_w_masks
    
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
    
    def add_bitmasks(self, gt_instances, im_h, im_w):
        for per_im_gt_inst in gt_instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)
                
                if len(per_im_bitmasks) > 0:
                    per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                    per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)

            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full
        
        
    def inference(self, _box_cls, _box_pred, image_sizes, _locations, _fpn_levels, _top_feat=None):
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
        locations = _locations.flatten(2)
        fpn_levels = _fpn_levels.flatten(2)
        
        if _top_feat is not None:
            top_feat = _top_feat.flatten(2)
        
        assert len(box_cls) == len(image_sizes)
        results = []
        
        scores = torch.sigmoid(box_cls)

        for i, (scores_per_image, box_pred_per_image, locations_per_image, fpn_levels_per_image, image_size) in enumerate(zip(
                scores, box_pred, locations, fpn_levels, image_sizes
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

            topk_loc_cat = locations_per_image[:, topk_inds_cat.reshape(-1)]
            topk_loc = topk_loc_cat[:, topk_inds]
            locations_per_image = topk_loc.transpose(0, 1)
            
            topk_fpn_level_cat = fpn_levels_per_image[:, topk_inds_cat.reshape(-1)]
            topk_fpn_level = topk_fpn_level_cat[:, topk_inds]
            fpn_levels_per_image = topk_fpn_level.transpose(0, 1)
            
            if _top_feat is not None:
                top_feat_per_image = top_feat[i]
                topk_top_feat_cat = top_feat_per_image[:, topk_inds_cat.reshape(-1)]
                topk_top_feat = topk_top_feat_cat[:, topk_inds]
                top_feat_per_image = topk_top_feat.transpose(0, 1)
            
            if self.nms:
                keep = batched_nms(box_pred_per_image, 
                                   scores_per_image, 
                                   labels_per_image, 
                                   0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                locations_per_image = locations_per_image[keep]
                fpn_levels_per_image = fpn_levels_per_image[keep]

                if _top_feat is not None:
                    top_feat_per_image = top_feat_per_image[keep]
                    
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.locations = locations_per_image
            result.fpn_levels = fpn_levels_per_image            
                        
            if _top_feat is not None:
                result.top_feat = top_feat_per_image
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

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)
        
        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
                        
            if results.pred_global_masks.shape[0] > 0:
                pred_global_masks = aligned_bilinear(
                    results.pred_global_masks, factor
                )
            else:
                # no instance will cause error when padding
                pred_global_masks = F.interpolate(
                results.pred_global_masks,
                size=(mask_h*factor, mask_w*factor),
                mode="bilinear", align_corners=False
            )
            
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results
