#!/usr/bin/env bash

CONFIG=$1
PRETRAIN=$2
GPUS=$3
PreWKD=$4
PORT=${PORT:-29500}
WORK_DIR="$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")/"
AWORK_DIR="${PreWKD}/${WORK_DIR}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT  \
    $(dirname "$0")/analysis_tools/svd_covariance.py $CONFIG --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
     --work_dir $AWORK_DIR --seed 0  --launcher pytorch ${@:5}
