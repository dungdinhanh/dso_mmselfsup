#!/usr/bin/env bash

USER=$1
CONFIG=$2
GPUS=$3
PORT=${PORT:-29501}
PYTHON=${PYTHON:-"/home/users/${USER}/.conda/envs/openss/bin/python3.8"}
WORK_DIR="$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")/"
AWORK_DIR="../../scratch/dso/openss/${WORK_DIR}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_kd_readminlog.py $CONFIG --work_dir $AWORK_DIR --seed 0 --launcher pytorch ${@:4}