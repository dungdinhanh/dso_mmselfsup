#!/usr/bin/env bash

bash tools/dist_svd_project_diff.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_poswneg_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--work_dir visualization/diff/poswneg

bash tools/dist_svd_project_diff.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam/simsiam_resnet18_2xb128-coslr-200e_in30p/epoch_200.pth \
--work_dir visualization/diff/simsiam

bash tools/dist_svd_project_diff.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simdis/simdis_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--work_dir visualization/diff/simdis

bash tools/dist_svd_project_diff.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--work_dir visualization/diff/pos

bash tools/dist_svd_project_diff.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_wneg_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
--work_dir visualization/diff/neg


bash tools/dist_svd_project_diff.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam/simsiam_resnet50_4xb64-coslr-200e_in30p/epoch_200.pth \
--work_dir visualization/diff/simsiam_r50



