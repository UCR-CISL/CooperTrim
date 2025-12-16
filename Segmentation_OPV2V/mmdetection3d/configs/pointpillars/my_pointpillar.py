auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
class_names = [
    'car',
    'truck',
    'bus',
    'emergency_vehicle',
    'other_vehicle',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'animal',
]
data_prefix = dict(img='', pts='v1.01-train/lidar', sweeps='v1.01-train/lidar')
data_root = 'data/lyft/'
dataset_type = 'LyftDataset'
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(backend_args=None, sweeps_num=10, type='LoadPointsFromMultiSweeps'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(
    use_camera=False,
    use_external=False,
    use_lidar=True,
    use_map=False,
    use_radar=False)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.001
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            max_voxels=(
                60000,
                60000,
            ),
            point_cloud_range=[
                -100,
                -100,
                -5,
                100,
                100,
                3,
            ],
            voxel_size=[
                0.25,
                0.25,
                8,
            ])),
    pts_backbone=dict(
        in_channels=64,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            2,
            2,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='naiveSyncBN2d'),
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    pts_bbox_head=dict(
        anchor_generator=dict(
            custom_values=[],
            ranges=[
                [
                    -100,
                    -100,
                    -1.8,
                    100,
                    100,
                    -1.8,
                ],
            ],
            reshape_out=True,
            rotations=[
                0,
                1.57,
            ],
            scales=[
                1,
                2,
                4,
            ],
            sizes=[
                [
                    2.5981,
                    0.866,
                    1.0,
                ],
                [
                    1.7321,
                    0.5774,
                    1.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    0.4,
                    0.4,
                    1,
                ],
            ],
            type='AlignedAnchor3DRangeGenerator'),
        assigner_per_size=False,
        bbox_coder=dict(code_size=7, type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=9,
        type='Anchor3DHead',
        use_direction_classifier=True),
    pts_middle_encoder=dict(
        in_channels=64, output_shape=[
            800,
            800,
        ], type='PointPillarsScatter'),
    pts_neck=dict(
        act_cfg=dict(type='ReLU'),
        in_channels=[
            64,
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='naiveSyncBN2d'),
        num_outs=3,
        out_channels=256,
        start_level=0,
        type='mmdet.FPN'),
    pts_voxel_encoder=dict(
        feat_channels=[
            64,
        ],
        in_channels=4,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='naiveSyncBN1d'),
        point_cloud_range=[
            -100,
            -100,
            -5,
            100,
            100,
            3,
        ],
        type='HardVFE',
        voxel_size=[
            0.25,
            0.25,
            8,
        ],
        with_cluster_center=True,
        with_distance=False,
        with_voxel_center=True),
    test_cfg=dict(
        pts=dict(
            max_num=500,
            min_bbox_size=0,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            use_rotate_nms=True)),
    train_cfg=dict(
        pts=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.6,
                type='Max3DIoUAssigner'),
            code_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            debug=False,
            pos_weight=-1)),
    type='MVXFasterRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            20,
            23,
        ],
        type='MultiStepLR'),
]
point_cloud_range = [
    -100,
    -100,
    -5,
    100,
    100,
    3,
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='lyft_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='v1.01-train/lidar', sweeps='v1.01-train/lidar'),
        data_root='data/lyft/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'bus',
            'emergency_vehicle',
            'other_vehicle',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'animal',
        ]),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                sweeps_num=10,
                type='LoadPointsFromMultiSweeps'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -100,
                            -100,
                            -5,
                            100,
                            100,
                            3,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='LyftDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='lyft_infos_val.pkl',
    backend_args=None,
    data_root='data/lyft/',
    metric='bbox',
    type='LyftMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(backend_args=None, sweeps_num=10, type='LoadPointsFromMultiSweeps'),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -100,
                    -100,
                    -5,
                    100,
                    100,
                    3,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=24)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='lyft_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='v1.01-train/lidar', sweeps='v1.01-train/lidar'),
        data_root='data/lyft/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'bus',
            'emergency_vehicle',
            'other_vehicle',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'animal',
        ]),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                sweeps_num=10,
                type='LoadPointsFromMultiSweeps'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                rot_range=[
                    -0.3925,
                    0.3925,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -100,
                    -100,
                    -5,
                    100,
                    100,
                    3,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -100,
                    -100,
                    -5,
                    100,
                    100,
                    3,
                ],
                type='ObjectRangeFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='LyftDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(backend_args=None, sweeps_num=10, type='LoadPointsFromMultiSweeps'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        rot_range=[
            -0.3925,
            0.3925,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTrans'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        point_cloud_range=[
            -100,
            -100,
            -5,
            100,
            100,
            3,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -100,
            -100,
            -5,
            100,
            100,
            3,
        ],
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='lyft_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='v1.01-train/lidar', sweeps='v1.01-train/lidar'),
        data_root='data/lyft/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'bus',
            'emergency_vehicle',
            'other_vehicle',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'animal',
        ]),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                sweeps_num=10,
                type='LoadPointsFromMultiSweeps'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -100,
                            -100,
                            -5,
                            100,
                            100,
                            3,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='LyftDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='lyft_infos_val.pkl',
    backend_args=None,
    data_root='data/lyft/',
    metric='bbox',
    type='LyftMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.25,
    0.25,
    8,
]
work_dir = './work_dirs/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d-range100'
