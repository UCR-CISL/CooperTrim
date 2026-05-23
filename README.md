# COOPERTRIM: Adaptive Data Selection For Uncertainty-Aware Cooperative Perception 

Official Pytorch Implementation of the framework **COOPERTRIM** proposed in our paper [**COOPERTRIM: Adaptive Data Selection For Uncertainty-Aware Cooperative Perception**](https://openreview.net/pdf?id=8NgKNuHRiH) accepted by **ICLR2026**.

[![paper](https://img.shields.io/badge/OpenReview-Paper-blue.svg)](https://openreview.net/pdf?id=8NgKNuHRiH)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://cisl.ucr.edu/CooperTrim)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/cisl-hf/CooperTrim)

<div align="center">
  <img src="images/System_architecture.png" width="600"/>
</div>

## Overview

We present <strong>COOPERTRIM</strong>, an adaptive feature selection framework in cooperative perception, which enhances representation learning through temporal uncertainty-driven feature selection for bandwidth-efficient, accurate perception in multi-agent systems. It addresses key challenges of relevance, identifying the most impactful features for downstream tasks, and quantity, determining the optimal point to stop sharing based on scene and task complexity. We employed an ϵ-greedy training method that optimizes the bandwidth-performance balance by facilitating effective exploration and exploitation during training.

CooperTrim is compatible with multiple intermediate fusion models — CoBEVT, AttFuse, DiscoNet, F-Cooper, SwissCheese, and Where2comm — without modifying their underlying architectures.

<p align="center">
<img src="images/Adaptive_selection.png" width="600" alt="">
</p>

<p align="center">
<img src="images/Performance_improvement.png" width="600" alt="">
</p>

COOPERTRIM adaptively requests data based on scene complexity. Dynamic objects trigger higher request volumes (Frames 1200, 200, 1700), as do complex static elements like intersections (Frames 900, 250, 1600). Solid green lines indicate CooperTrim maintains high IoU despite reduced bandwidth compared to baseline CoBEVT (dashed green lines).



## Getting Started

### Data Preparation

**OPV2V:** Download from [UCLA BOX](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V). For large files, use the chunked downloads and merge:
```bash
cat train.zip.part* > train.zip
unzip train.zip
```
See [our website](https://mobility-lab.seas.ucla.edu/opv2v/) for dataset details.

**V2V4Real:** Download from the [V2V4Real website](https://research.seas.ucla.edu/mobility-lab/v2v4real/) (OPV2V format). Organize as:
```
v2v4real/
├── train/
│   └── testoutput_CAV_data_2022-03-15-09-54-40_1/
├── validate/
└── test/
```

### Installation

```bash
git clone https://github.com/UCR-CISL/CooperTrim.git
cd CooperTrim
```

Go to the sub-folder for your task: `Segmentation_OPV2V`, `3D_Detection_OPV2V`, or `3D_Detection_V2V4Real`.

```bash
# Segmentation_OPV2V:
conda env create -f cobevt_env.yaml
conda activate cobevt_env

# 3D_Detection_OPV2V or 3D_Detection_V2V4Real:
conda env create -f opencood_env.yaml
conda activate opencood_env
```

Then compile and install:
```bash
python setup.py build_ext --inplace
python setup.py develop
```

### Visualization

```bash
cd CooperTrim
python Segmentation_OPV2V/opv2v/opencood/visualization/visualize_data.py [--scene ${SCENE_NUMBER} --sample ${SAMPLE_NUMBER}]
```

- `--scene`: scene index (default: 4)
- `--sample`: sample index within scene (default: 10)

### Training

Before training, copy the desired config from `configs/` into your checkpoint folder and rename it `config.yaml`. CooperTrim configs are in `configs/Segmentation_OPV2V/`, `configs/3D_Detection_OPV2V/`, and `configs/3D_Detection_V2V4Real/`.

**Single GPU:**
```bash
# Segmentation
cd CooperTrim/Segmentation_OPV2V/opv2v/
python opencood/tools/train_camera.py --hypes_yaml opencood/checkpoints_test/config.yaml

# Detection
cd CooperTrim/3D_Detection_OPV2V
python opencood/tools/train.py --hypes_yaml opencood/ckp_test/config.yaml --model_dir opencood/ckp_test [--half]
```

**Multiple GPUs:**
```bash
# Segmentation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 opencood/tools/train_camera.py --hypes_yaml opencood/checkpoints_test/config.yaml --model_dir opencood/checkpoints_test

# Detection
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 opencood/tools/train.py --hypes_yaml opencood/ckp_test/config.yaml --model_dir opencood/ckp_test
```

### Testing

```bash
# Segmentation
cd CooperTrim/Segmentation_OPV2V/opv2v/
python opencood/tools/inference_camera.py --model_dir opencood/checkpoints_test [--model_type static]

# Detection
cd CooperTrim/3D_Detection_OPV2V
python opencood/tools/inference.py --model_dir opencood/checkpoints_test --fusion_method intermediate
```

Evaluation results are saved in the model directory.

## Pretrained Checkpoints

All checkpoints are hosted on [HuggingFace](https://huggingface.co/cisl-hf/CooperTrim). Download and place each `.pth` file in the same folder as the corresponding `config.yaml`.

`_dyn` = dynamic object target; `_st` = static object target.

### Segmentation (OPV2V, BEV semantic segmentation)

| Checkpoint | Base Model | Target | Link |
|---|---|---|---|
| `baseline_cobevt_dyn.pth` | CoBEVT | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_cobevt_dyn.pth) |
| `baseline_cobevt_st.pth` | CoBEVT | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_cobevt_st.pth) |
| `baseline_attfuse_dyn.pth` | AttFuse | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_attfuse_dyn.pth) |
| `baseline_attfuse_st.pth` | AttFuse | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_attfuse_st.pth) |
| `baseline_disconet_dyn.pth` | DiscoNet | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_disconet_dyn.pth) |
| `baseline_disconet_st.pth` | DiscoNet | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_disconet_st.pth) |
| `baseline_fcooper_dyn.pth` | F-Cooper | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_fcooper_dyn.pth) |
| `baseline_fcooper_st.pth` | F-Cooper | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_fcooper_st.pth) |
| `baseline_swisscheese_dyn.pth` | SwissCheese | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_swisscheese_dyn.pth) |
| `baseline_swisscheese_st.pth` | SwissCheese | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_swisscheese_st.pth) |
| `baseline_where2comm_dyn.pth` | Where2comm | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_where2comm_dyn.pth) |
| `baseline_where2comm_st.pth` | Where2comm | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/baseline_where2comm_st.pth) |
| `coopertrim_cobevt_dyn.pth` | CoBEVT + **CooperTrim** | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/coopertrim_cobevt_dyn.pth) |
| `coopertrim_cobevt_st.pth` | CoBEVT + **CooperTrim** | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/coopertrim_cobevt_st.pth) |
| `coopertrim_attfuse_dyn.pth` | AttFuse + **CooperTrim** | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/coopertrim_attfuse_dyn.pth) |
| `coopertrim_attfuse_st.pth` | AttFuse + **CooperTrim** | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/coopertrim_attfuse_st.pth) |
| `coopertrim_disconet_dyn.pth` | DiscoNet + **CooperTrim** | Dynamic | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/coopertrim_disconet_dyn.pth) |
| `coopertrim_disconet_st.pth` | DiscoNet + **CooperTrim** | Static | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/coopertrim_disconet_st.pth) |

### Detection (3D object detection)

| Checkpoint | Dataset | Method | Link |
|---|---|---|---|
| `det_opv2v_baseline_cobevt.pth` | OPV2V | CoBEVT baseline | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/det_opv2v_baseline_cobevt.pth) |
| `det_opv2v_coopertrim.pth` | OPV2V | **CooperTrim** | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/det_opv2v_coopertrim.pth) |
| `det_v2v4real_baseline_cobevt.pth` | V2V4Real | CoBEVT baseline | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/det_v2v4real_baseline_cobevt.pth) |
| `det_v2v4real_coopertrim.pth` | V2V4Real | **CooperTrim** | [download](https://huggingface.co/cisl-hf/CooperTrim/resolve/main/det_v2v4real_coopertrim.pth) |

## Citation

If you use CooperTrim in your research, please cite:
```bibtex
@inproceedings{mukhopadhyaycoopertrim,
  title={CooperTrim: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception},
  author={Mukhopadhyay, Shilpa and Roy-Chowdhury, Amit and Qiu, Hang},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
