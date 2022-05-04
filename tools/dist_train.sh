#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
WORK_DIR="$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")/"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --seed 0 --work_dir $WORK_DIR  --launcher pytorch ${@:3}
