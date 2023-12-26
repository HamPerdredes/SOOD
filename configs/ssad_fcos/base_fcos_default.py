from configs.rotated_fcos.rotated_fcos_r50_fpn_le90 import model as detector
from configs.rotated_fcos.rotated_fcos_r50_fpn_le90 import angle_version
# if need to replace other dataset, just modify imports below.
from configs._base_.datasets.dotav15_val import img_norm_cfg, dataset_type
from configs._base_.datasets.dotav15_val import data as src_data
import torchvision.transforms as transforms
from copy import deepcopy

_base_ = [
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

custom_imports = dict(
    imports=['ssad'],
    allow_failed_imports=False)

# following Dense Teacher, we apply weak & strong augmentation for both labeled and unlabeled data
# weak_pipeline is the same as DOTA_V1's train pipeline
# dense teacher manner pipeline.
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="sup_weak"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
common_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
strong_pipeline = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply', operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply', operations=[
        dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])
    ]),
    # dict(type='DTRandCrop'),
    dict(type='DTToNumpy'),
    dict(type="ExtraAttrs", tag="unsup_strong"),
]
weak_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="unsup_weak"),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="LoadEmptyAnnotations", with_bbox=True),
    dict(type="STMultiBranch", unsup_strong=deepcopy(strong_pipeline), unsup_weak=deepcopy(weak_pipeline),
         common_pipeline=common_pipeline, is_seq=True),
]

data = dict(
    samples_per_gpu=None,
    workers_per_gpu=None,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type=dataset_type,
            ann_file=None,
            img_prefix=None,
            pipeline=sup_pipeline,
        ),
        unsup=dict(
            type=dataset_type,
            ann_file=None,
            img_prefix=None,
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=src_data['val'],
    test=src_data['test'],
    sampler=dict(
        train=dict(
            type="GroupMultiSourceSampler",
            sample_ratio=[2, 1]
        )
    ),
)

# todo: fix hard-code here
detector['type'] = 'SemiRotatedFCOS'
detector['bbox_head']['type'] = 'SemiRotatedFCOSHead'
detector['bbox_head']['num_classes'] = 16
model = dict(
    type="RotatedSSTGDenseTeacher",
    model=detector,
    semi_loss=dict(type='RotatedSingleStageDTLoss'),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=6400,
        sup_weight=1.0,
        unsup_weight=1.0,
        weight_suppress="linear",
        logit_specific_weights=dict(),
        region_ratio=0.01
    ),
    test_cfg=dict(inference_on="teacher"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9996, interval=1, start_steps=3200),
]
evaluation = dict(type="SubModulesDistEvalHook", interval=3200, metric='mAP',
                  save_best='mAP')
# some config as in the manuscript
lr_config = dict(step=[120000, 160000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=3200, max_keep_ckpts=2)

# Default: disable fp16 training
# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="ssad_fcos",
                name="Default",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
