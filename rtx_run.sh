ilod='[16, 20]'
optim='Adam'
lr_weight=1e-4
lr_mask=1e-5
model='pb[body][:3]'
out="voc${ilod}_${model}_${optim}:${lr_weight}_${lr_mask}"
port=$RANDOM&10000
python -m torch.distributed.launch --nproc_per_node=4 --master_port ${port} train.py \
--data-path /root/userfolder/data/voc2007 \
--dataset voc2007 --ilod "${ilod}" --num-classes 21 \
--base-model models/resnet50-19c8e357.pth \
--pb body --freeze None \
--epochs 24 --lr-steps 16 22 -b 2 \
--lr-w ${lr_weight} --lr-m ${lr_mask} --optim ${optim} \
--output-dir "${out}" \
--mask-init 1s --mask-scale 1e-2 > $out'.out'