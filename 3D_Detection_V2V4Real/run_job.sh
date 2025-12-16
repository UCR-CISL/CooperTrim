#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=smukh039@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="V4PreAuto"
#SBATCH -p cisl
#SBATCH --gres=gpu:2
#SBATCH --wait-all-nodes=1
#SBATCH --output=output_%j-%N.txt

hostname
date
# # Activate Conda
source /home/csgrad/smukh039/miniforge3/etc/profile.d/conda.sh
conda activate opencood_env
which python


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2     /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra/config.yaml     --model_dir /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra > run_job_idtra_checkpoints_sub100_req_output.log 2>&1

# /home/csgrad/smukh039/miniforge3/envs/opencood_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/OpenCOOD_AutoNet/opencood/tools/train.py     --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/OpenCOOD_AutoNet/opencood/checkpoints_test/corpbevtconfig.yaml     --model_dir /home/csgrad/smukh039/AutoNetworkingRL/OpenCOOD_AutoNet/opencood/checkpoints_test --half > run_orig.log 2>&1
# /home/csgrad/smukh039/miniforge3/envs/opencood_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/OpenCOOD_AutoNet/opencood/tools/train.py     --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/OpenCOOD_AutoNet/opencood/ckp_cobevt_autonet/config.yaml     --model_dir /home/csgrad/smukh039/AutoNetworkingRL/OpenCOOD_AutoNet/opencood/ckp_cobevt_autonet > run_cobevt_l5.log 2>&1
# /home/csgrad/smukh039/miniforge3/envs/opencood_env/bin/python opencood/tools/train.py --hypes_yaml /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_baseline/config.yaml --model_dir  /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_baseline --half
#  /home/csgrad/smukh039/miniforge3/envs/opencood_env/bin/python  /home/csgrad/smukh039/AutoNetworkingRL/V2V4Real/opencood/tools/train.py --hypes_yaml /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_baseline_fulldata/config.yaml --model_dir  /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_baseline_fulldata > /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_baseline_fulldata.log 2>&1

#FT
#  /home/csgrad/smukh039/miniforge3/envs/opencood_env/bin/python  /home/csgrad/smukh039/AutoNetworkingRL/V2V4Real/opencood/tools/train.py --hypes_yaml /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_ft/config.yaml --model_dir  /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_ft > /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_ft.log 2>&1

#provided Pretrained+Autonet
/home/csgrad/smukh039/miniforge3/envs/opencood_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/V2V4Real/opencood/tools/train.py --hypes_yaml /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_pretrained_autonet_fulldata2/config.yaml --model_dir  /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_pretrained_autonet_fulldata2 > /data/HangQiu/proj/AutoNetSelection/v2vr_det_cobevt_pretrained_autonet_fulldata2.log 2>&1

