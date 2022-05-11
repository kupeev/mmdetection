# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.core import bbox_overlaps, multi_apply, reduce_mean
from mmdet.models.dense_heads import LDHead

from ..builder import HEADS, build_loss
from .gfl_head import GFLHead


@HEADS.register_module()
class LDHeadDouble(LDHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_ld=dict(
                     type='LocalizationDistillationLoss',
                     loss_weight=0.25,
                     T=10),
                 **kwargs):

        super(LDHeadDouble, self).__init__(num_classes, in_channels, loss_ld = loss_ld, **kwargs)
        self.var = 20

    def forward_train(self,
                      x,
                      out_teacher,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """

        """
            if self is instance of LDHead
                outs = {tuple: 2}
                0 = {list: 5}
                    0 = {Tensor: (1, 5, 100, 100)}
                    ...
                    4 = {Tensor: (1, 5, 7, 7)}

                1 = {list: 5}
                    0 = {Tensor: (1, 68, 100, 100)}
                    ...
                    4 = {Tensor: (1, 68, 7, 7)}
            if self is instance of LDHeadDouble
                outs = {tuple: 2}

                0 = {list: 5}
                    0 = {Tensor: (1, 5, 100, 100)}
                    ...
                    4 = {Tensor: (1, 5, 7, 7)}

                1 = {list: 5}
                    0 = {Tensor: (1, 68, 100, 100)}
                    ...
                    4 = {Tensor: (1, 68, 7, 7)}
        """

        outs = self(x)
        # qq
        soft_target = out_teacher[1]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, soft_target, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, soft_target, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
