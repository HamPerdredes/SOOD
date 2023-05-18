_base_ = "base_fcos_default.py"

data_root = 'data/split_ss_dota_v15/'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        sup=dict(
            ann_file=data_root + 'train_10_labeled/annfiles/',
            img_prefix=data_root + 'train_10_labeled/images/',
            classes=classes,
        ),
        unsup=dict(
            ann_file=data_root + 'train_10_unlabeled/empty_annfiles/',
            img_prefix=data_root + 'train_10_unlabeled/images/',
            classes=classes,
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[2, 1],
        )
    ),
)

model = dict(
    semi_loss=dict(type='RotatedSingleStageDTLoss', loss_type='pr_origin_p5',
                   cls_loss_type='bce', dynamic_weight='50ang',
                   aux_loss='ot_loss_norm', aux_loss_cfg=dict(clamp_ot=True)),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=6400,
    )
)

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="rotated_DenseTeacher_10percent",
        #         name="default_bce4cls",
        #     ),
        #     by_epoch=False,
        # ),
    ],
)
