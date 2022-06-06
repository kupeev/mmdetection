# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

import mmcv
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import matplotlib.pyplot as plt
#from mmdet.apis import inference_detector, show_result_pyplot #leads to cyclic dependance
from demo.davidk.general_dk import global_vars


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 teacher_config=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head) ## qq old

        if teacher_config and bbox_head.type == 'LDHeadDouble':
            assert 'LDHeadDouble' in str(type(self.bbox_head))
            teacher_config = mmcv.Config.fromfile(teacher_config)
            bbox_head_student = teacher_config['model']['bbox_head']
            bbox_head_student.update(train_cfg=train_cfg)
            bbox_head_student.update(test_cfg=test_cfg)
            self.bbox_head.bbox_head_student = build_head(bbox_head_student)


        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        if 0:
            from mmdet.apis import inference_detector, show_result_pyplot
            img = '/home/konstak/data/mmdet/data/train/10.jpg'

            config_file = '/home/konstak/projects2/mmdetection/configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
            config = mmcv.Config.fromfile(config_file)
            self.cfg = config  # save the config in the model for convenience

            result = inference_detector(self, img)

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

        if global_vars.pars.N1:
            assert self.bbox_head.__class__.__name__ in ['LDHeadDouble', 'LDHead']

            feat_N1 = self.teacher_model.extract_feat(img)
            results_list_N1 = self.teacher_model.bbox_head.simple_test(
                feat_N1, img_metas, rescale=rescale)
            bbox_results_N1 = [
                bbox2result(det_bboxes, det_labels, self.teacher_model.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list_N1
            ]

            from mmdet.apis import show_result_pyplot
            show_result_pyplot(self.teacher_model, img_metas[0]['filename'], bbox_results_N1[0],\
                out_file = global_vars.pars.out_dir + 'res.' + 'N1.' + str(self.teacher_model.__class__.__name__ ) + '.teacher_model.'
                           + str(global_vars.cnt) + '.png', fix_imshow_det_bboxes = 1)
            global_vars.cnt += 1
            plt.close()
        #if global_vars.pars.N1:
        if global_vars.pars.N2:
            assert self.bbox_head.__class__.__name__ in ['LDHeadDouble', 'LDHead']

            from mmdet.apis import show_result_pyplot
            shorten_class_name = 'KD' if str(self.__class__.__name__) == 'KnowledgeDistillationSingleStageDetector' \
                    else str(self.__class__.__name__)

            if global_vars.cnt % 10 == 0:
                show_result_pyplot(self, img_metas[0]['filename'], bbox_results[0],\
                    out_file = global_vars.pars.out_dir + 'res.' + shorten_class_name + '.N2_trained_model.'
                               + str(global_vars.cnt) + '.png', fix_imshow_det_bboxes = 1)
            global_vars.cnt += 1
            plt.close()
        #if global_vars.pars.N1:

        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
