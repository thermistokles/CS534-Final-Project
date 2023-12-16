#!/bin/bash
#SBATCH -N 1
#SBATCH -n 25
#SBATCH --mem=24g
#SBATCH -J "SIIM-ACR Pneumothorax ConvNextTiny"
#SBATCH -p short
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100|A30

echo "Your script has started." | mail -s "Script started" mamcinerney@wpi.edu

module load python/3.8.13/slu6jvw
module load cuda11.2/blas/11.2.2
module load cuda11.2/fft/11.2.2
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33

export set XLA_FLAGS=--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.2/toolkit/11.2.2

python cvntiny.py

echo "I am done"
sbatch run-cvnsmall.sh
