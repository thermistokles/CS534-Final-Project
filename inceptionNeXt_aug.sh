#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=12g
#SBATCH -J "SIIM-ACR Pneumothorax InceptionNeXt"
#SBATCH -p long
#SBATCH -t 2-23:00:00
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100|A30

module load cuda11.2/blas/11.2.2
module load cuda11.2/fft/11.2.2
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33

export set XLA_FLAGS=--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.2/toolkit/11.2.2


python inceptionNeXt_aug.py