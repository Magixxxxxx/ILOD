#!/bin/bash

#SBATCH -o slurm.%j.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu4Q
#SBATCH --qos=gpuq

cd /root/zjw/ILOD-FasterRCNN
python train.py