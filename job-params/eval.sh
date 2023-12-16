#!/bin/bash
#SBATCH -N 1
#SBATCH -n 25
#SBATCH --mem=24g
#SBATCH -J "[EVAL] ConvNext Models"
#SBATCH -p long
#SBATCH -t 2-23:00:00
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100|A30

module load python/3.8.13/slu6jvw
module load cuda11.2/blas/11.2.2
module load cuda11.2/fft/11.2.2
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33

export set XLA_FLAGS=--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.2/toolkit/11.2.2

python evaluate.py

echo "I am done"
