#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=smukh039@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="stSwissCheese"
#SBATCH -p cisl
#SBATCH --gres=gpu:2
#SBATCH --wait-all-nodes=1
#SBATCH --output=output_%j-%N.txt

hostname
date
# sleep 10800
# # Activate Conda
source /home/csgrad/smukh039/miniforge3/etc/profile.d/conda.sh
conda activate cobevt_env
which python
export PYTHONPATH=/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v:$PYTHONPATH



# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2     /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra/config.yaml     --model_dir /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra > run_job_idtra_checkpoints_sub100_req_output.log 2>&1

/home/csgrad/smukh039/miniforge3/envs/cobevt_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /data/HangQiu/proj/AutoNetSelection/checkpoints_baseline_swisscheese_static/config.yaml     --model_dir /data/HangQiu/proj/AutoNetSelection/checkpoints_baseline_swisscheese_static > run_job_baseline_swisscheese_st.log 2>&1
# /home/csgrad/smukh039/miniforge3/envs/cobevt_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /data/HangQiu/proj/AutoNetSelection/checkpoints_cobevt_cp_curriculum_dyn_epsilon_from_base2/config.yaml     --model_dir /data/HangQiu/proj/AutoNetSelection/checkpoints_cobevt_cp_curriculum_dyn_epsilon_from_base2 > run_job_cobevt_cp_curriculum_EG_from_base_dyn2.log 2>&1 #cp_stage2_st_loss.log 2>&1

# /home/csgrad/smukh039/miniforge3/envs/cobevt_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /data/HangQiu/proj/AutoNetSelection/checkpoints_swapfuse_st10/config.yaml     --model_dir /data/HangQiu/proj/AutoNetSelection/checkpoints_swapfuse_st10 > run_job_swapfuse_st10.log 2>&1