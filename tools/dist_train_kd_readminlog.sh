#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
WORK_DIR="$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")/"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
     $(dirname "$0")/train_kd_readminlog.py $CONFIG --work_dir $WORK_DIR --seed 0 --launcher pytorch ${@:3}
