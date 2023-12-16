#!/bin/bash
#SBATCH -N 1
#SBATCH -n 15
#SBATCH --mem=32g
#SBATCH -J "RepVitJob"
#SBATCH -p long
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:4
#SBATCH -C A100|V100
module load cuda11.6/blas/
module load cuda11.6/fft/
module load cuda11.6/toolkit/
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12346 --use_env RepViTScript.py
