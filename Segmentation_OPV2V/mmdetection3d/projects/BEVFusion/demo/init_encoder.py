from mmengine import Config
#shilpa apptainer
import sys
sys.path.append('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/mmdetection3d')
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

def build_encoder(config='/home/shilpa/autoRL/CoBEVT_AutoNet/mmdetection3d/projects/BEVFusion/configs/bevfusion_encoder.py', device='cuda:0'):
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    return model

def build_transfusionhead(config='/home/shilpa/autoRL/CoBEVT_AutoNet/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar180.py', device='cuda:0'):
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model['bbox_head'])
    return model

def build_bevfusion_model(config='/home/shilpa/autoRL/CoBEVT_AutoNet/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar180.py', device='cuda:0'):
    register_all_modules()
    cfg = Config.fromfile(config)
    backbone = MODELS.build(cfg.model)
    head = MODELS.build(cfg.model['bbox_head'])
    return backbone, head


if __name__ == '__main__':
    model = build_encoder()
    print(model)