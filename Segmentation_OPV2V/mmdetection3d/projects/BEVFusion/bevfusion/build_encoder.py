from mmengine import Config
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

def build_encoder(config='/home/jerryli/MultiModalityPerception/mmdetection3d/projects/BEVFusion/configs/bevdusion_encoder.py', device='cuda:0'):
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    return model

if __name__ == '__main__':
    model = init_encoder()
    print(model)