#!/usr/bin/env bash

bash tools/dist_svd_project_diff_multiview.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_poswneg_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--dataset_config configs/benchmarks/classification/svd_multiview_imagenet.py \
--work_dir visualization/diff_mv/poswneg

bash tools/dist_svd_project_diff_multiview.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam/simsiam_resnet18_2xb128-coslr-200e_in30p/epoch_200.pth \
--dataset_config configs/benchmarks/classification/svd_multiview_imagenet.py \
--work_dir visualization/diff_mv/simsiam

bash tools/dist_svd_project_diff_multiview.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simdis/simdis_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--dataset_config configs/benchmarks/classification/svd_multiview_imagenet.py \
--work_dir visualization/diff_mv/simdis

bash tools/dist_svd_project_diff_multiview.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--dataset_config configs/benchmarks/classification/svd_multiview_imagenet.py \
--work_dir visualization/diff_mv/pos

bash tools/dist_svd_project_diff_multiview.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_wneg_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--dataset_config configs/benchmarks/classification/svd_multiview_imagenet.py \
--work_dir visualization/diff_mv/neg


bash tools/dist_svd_project_diff_multiview.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam/simsiam_resnet50_4xb64-coslr-200e_in30p/epoch_200.pth \
--dataset_config configs/benchmarks/classification/svd_multiview_imagenet.py \
--work_dir visualization/diff_mv/simsiam_r50



