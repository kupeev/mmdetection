#based on http://localhost:8888/notebooks/demo/inference_demo.ipynb
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

if 0: #from demo
    config_file = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = '../../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
elif 1:
    # just for ld teacher
    #from ld_r18_gflv1_r101_fpn_coco_1x_double.py
    #   teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa
    #   teacher_config = '/home/konstak/projects2/mmdetection/configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py',
    checkpoint_file = '/home/konstak/Downloads/_gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
    config_file = '/home/konstak/projects2/mmdetection/configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
elif 0: #our teach student
    config_file = '../../work_dirs/config_ld_double/_config_ld_double.py'
    #checkpoint_file = '../../work_dirs/config_ld_double/_epoch_1.pth'
    checkpoint_file = '../../work_dirs/config_ld_double/epoch_12.pth'
elif 0: #their teach student
    #... Checkpoints will be saved to /home/konstak/projects2/mmdetection/work_dirs/config_ld by HardDiskBackend.
    config_file = '../../work_dirs/config_ld/_config_ld.py'
    checkpoint_file = '../../work_dirs/config_ld/epoch_12.pth'


#--------------------------------
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
#--------------------------------
# test a single image
if 0:
    img = 'demo.jpg'
elif 1:
    img = '/home/konstak/data/mmdet/data/train/10.jpg'

result = inference_detector(model, img)
#--------------------------------
# show the results
show_result_pyplot(model, img, result)
#--------------------------------
tmp=10
#--------------------------------
#--------------------------------


