#!/bin/bash


cmd="bash tools/dist_train_kd_readminlog.sh \
configs/selfsup/simsiam_kd_test/simsiamkd_min10epoch_resnet18_2xb128-coslr-200e_in30p.py 2 \
--teacher_path work_dirs/min_loss_epochs.csv"
echo ${cmd}
eval ${cmd}

