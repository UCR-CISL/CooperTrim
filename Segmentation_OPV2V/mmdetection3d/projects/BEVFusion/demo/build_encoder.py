from mmengine import Config
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

def build_encoder(config='/home/shilpa/autoRL/CoBEVT_AutoNet/mmdetection3d/projects/BEVFusion/configs/bevfusion_encoder.py', device='cuda:0'):
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    return model

if __name__ == '__main__':
    model = build_encoder()
    print(model)