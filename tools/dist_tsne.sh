#!/usr/bin/env bash

CONFIG=$1
PRETRAIN=$2
GPUS=$3
PORT=${PORT:-29500}
WORK_DIR="$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")/"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT  \
    $(dirname "$0")/analysis_tools/visualize_tsne.py $CONFIG --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
     --work_dir $WORK_DIR --seed 0  --launcher pytorch ${@:4}
