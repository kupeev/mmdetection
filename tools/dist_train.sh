#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

"""
https://github.com/HikariTJU/LD

# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'

./tools/dist_train.sh configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py 8

      seed = init_random_seed(args.seed)
      work = default_pg.broadcast([tensor], opts)
      RuntimeErrorRuntimeError: RuntimeErrorRuntimeError: RuntimeErrorRuntimeErrorNCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:957, invalid usage, NCCL version 21.0.3
      ncclInvalidUsage: This usually reflects invalid usage of NCCL library (such as too many async ops, too many collectives at once, mixing streams in a group, etc).RuntimeError
      : : NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:957, invalid usage, NCCL version 21.0.3

we run:

  ./tools/dist_train.sh configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py 1

  nichush args to train:
train path_congig --launcher pytorch 1
"""
