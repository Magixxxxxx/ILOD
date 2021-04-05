#!/bin/bash

#SBATCH -o slurm.%j.out
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu4Q
#SBATCH --qos=gpuq

cd /root/zjw/ILOD-FasterRCNN

python -m torch.distributed.launch --nproc_per_node=2 train.py \
--data-path /home/zhaojiawei/Data/COCO2017 \
--epochs 20 --lr-steps [13, 16] --lr 0.02 \
--base-model model/fasterrcnn_resnet50_fpn_pretrained.pth