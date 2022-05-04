#!/usr/bin/env bash

USER=$1
CONFIG=$2
CHECKPOINT=$3
GPUS=$4
PORT=${PORT:-29500}
PYTHON=${PYTHON:-"/home/users/${USER}/.conda/envs/openss/bin/python3.8"}
WORK_DIR="$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")/"
AWORK_DIR="../../scratch/dso/openss/${WORK_DIR}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_kd.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5}
