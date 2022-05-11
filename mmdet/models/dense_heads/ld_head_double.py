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

