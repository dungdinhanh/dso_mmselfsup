#!/usr/bin/env bash

set -e
set -x

USER=$1
CFG=$2  # use cfgs under "configs/benchmarks/classification/imagenet/*.py"
PRETRAIN=$3  # pretrained model
GPUS=$4
PY_ARGS=${@:5}
# GPUS=${GPUS:-8}  # When changing GPUS, please also change imgs_per_gpu in the config file accordingly to ensure the total batch size is 256.
PORT=${PORT:-29500}
PYTHON=${PYTHON:-"/home/users/${USER}/.conda/envs/openss/bin/python3.8"}
# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"
AWORK_DIR="../../scratch/dso/openss/${WORK_DIR}"

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
    --work_dir $AWORK_DIR --seed 0 --launcher="pytorch" ${PY_ARGS}
