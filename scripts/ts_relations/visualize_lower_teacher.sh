#!/usr/bin/env bash


bash tools/dist_teacher_student_relation.sh configs/selfsup/visualization/simsiam_r18s_r50t.py 2 \
--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_poswneg_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--dataset_config configs/benchmarks/classification/svd_multiview_imagenet.py \
--work_dir visualization/ts_relations/