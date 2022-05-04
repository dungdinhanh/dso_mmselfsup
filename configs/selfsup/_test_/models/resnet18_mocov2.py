# model settings
model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=512,
        hid_channels=512,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
