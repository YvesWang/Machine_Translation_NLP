#!/bin/bash

#SBATCH --job-name=RNNGPU
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=55:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../MTNLPLog/mt_%j.out
module purge
#module load python3/intel/3.6.3
#module load jupyter-kernels/py3.5
source /home/tw1682/py36/bin/activate

CUDA_LAUNCH_BLOCKING=1 python -u train.py 
