# my_quick_run_4.py
# run main from train.py with passing config as arg
#following /home/konstak/projects2/mmdetection/tools/dist_train.sh

from tools import train

from train import main

#main('/home/konstak/projects2/mmdetection/configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py')
main()


tmp=10

"""

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

train path_congig --launcher pytorch 1

./tools/dist_train.sh configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py 1

"""








