# optimizer
optimizer = dict(type='SGD', lr=30, momentum=0.9, weight_decay=0.0)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
