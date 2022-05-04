_base_ = [
    '../_test_/models/simsiam_kd_test/resnet18_simsiam_dimcollapse_lpips2.py',
    '../_test_/datasets/imagenet1p1r_mocov2_wori_b64_cluster.py',
    '../_test_/schedules/sgd_coslr-200e_in30p_kd.py',
    '../_test_/default_runtime.py',
]

# model settings
model = dict(
    type='SimSiam_dimcollapsecheckLPIPS_fixedfirstview',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=1024,
        out_channels=2048,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)),
    teacher=dict(
                type='SimSiam',
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    in_channels=3,
                    out_indices=[4],  # 0: conv-1, x: stage-x
                    norm_cfg=dict(type='SyncBN'),
                    zero_init_residual=True),
                neck=dict(
                    type='NonLinearNeck',
                    in_channels=2048,
                    hid_channels=2048,
                    out_channels=2048,
                    num_layers=3,
                    with_last_bn_affine=False,
                    with_avg_pool=True),
                head=dict(
                    type='LatentPredictHead',
                    predictor=dict(
                        type='NonLinearNeck',
                        in_channels=2048,
                        hid_channels=512,
                        out_channels=2048,
                        with_avg_pool=False,
                        with_last_bn=False,
                        with_last_bias=True))
    ))

# set base learning rate
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

lr=0.003


# additional hooks
custom_hooks = [
    dict(type='SimSiamHook', priority='HIGH', fix_pred_lr=True, lr=lr)
]

# optimizer
optimizer = dict(lr=lr, paramwise_options={'predictor': dict(fix_lr=True)})


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)


# runtime settings
runner = dict(type='KDBasedRunnerSaveImagesAllLPIPS2', max_epochs=1)

load_from = "../../scratch/dso/openss/work_dirs/selfsup/simsiam_kd_test/simsiam_dimcollapse_resnet18/epoch_150.pth"

workflow = [('val', 1)]


