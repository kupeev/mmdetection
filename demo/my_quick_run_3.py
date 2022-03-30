# my_quick_run_3.py
# following https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html#
#
# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp

import mmcv

# The new config inherits a base config to highlight the necessary modification
_base_ = '/home/konstak/projects2/mmdetection/configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py'

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# here run in terminal
# python tools/train.py configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py

if 0:
    import mmcv
    import os
    import matplotlib.pyplot as plt
    img = mmcv.imread('kitti_tiny/training/image_2/000073.jpeg')
    plt.figure(figsize=(15, 10))
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()

import copy
import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


from mmcv import Config
cfg = Config.fromfile(_base_)


from mmdet.apis import set_random_seed


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


# does not work, thus use __init__
if 0:
    cfg.custom_imports = dict(imports=['mmdet.core.utils.my_hook'], allow_failed_imports=False)

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

# ./tools/dist_train.sh configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py 1