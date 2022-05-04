#!/usr/bin/env bash

#bash tools/dist_svd_project.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
#--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_poswneg_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
#--work_dir visualization/proj/poswneg
#
#bash tools/dist_svd_project.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
#--checkpoint work_dirs/selfsup/simsiam/simsiam_resnet18_2xb128-coslr-200e_in30p/epoch_200.pth \
#--work_dir visualization/proj/simsiam
#
#bash tools/dist_svd_project.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
#--checkpoint work_dirs/selfsup/simdis/simdis_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
#--work_dir visualization/proj/simdis
#
#bash tools/dist_svd_project.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
#--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
#--work_dir visualization/proj/pos
#
#bash tools/dist_svd_project.sh configs/selfsup/visualization/simsiam_resnet18_2xb128-coslr-200e_in30p.py 2 \
#--checkpoint work_dirs/selfsup/simsiam_kd/simsiamkd_wneg_resnet18_4xb64-coslr-200e_in30p/epoch_200.pth \
#--work_dir visualization/proj/neg


bash tools/dist_svd_project.sh configs/selfsup/visualization/simsiam_resnet50_2xb128-coslr-200e_in30p.py 2 \
--checkpoint work_dirs/selfsup/simsiam/simsiam_resnet50_4xb64-coslr-200e_in30p/epoch_200.pth \
--work_dir visualization/proj/simsiam_r50



