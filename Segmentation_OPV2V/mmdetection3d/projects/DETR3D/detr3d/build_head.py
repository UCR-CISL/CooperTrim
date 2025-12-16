from mmengine import Config
from mmdet3d.registry import MODELS

def build_head(config='/home/jerryli/MultiModalityPerception/mmdetection3d/projects/DETR3D/configs/detr3d_r101_gridmask.py', device='cuda:0'):
    if isinstance(config, dict):
        cfg = Config(config)
    elif isinstance(config, str):
        cfg = Config.fromfile(config['model']['args']['pts_bbox_head'])
    else:
        raise ValueError("Invalid type for `config`: expected dict or str")
    print(cfg)
    model = MODELS.build(cfg)
    return model

if __name__ == '__main__':
    model = init_encoder()
    print(model)