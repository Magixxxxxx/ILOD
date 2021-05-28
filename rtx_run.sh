ilod='[16, 20]'
optim='Adam'
lr_weight=1e-4
lr_mask=1e-5
model='pb[body]moco2002'
out="voc${ilod}_${model}_${optim}:${lr_weight}_${lr_mask}"
port=$(python -c 'import random;print(random.randrange(10000, 20000))')

python -m torch.distributed.launch --nproc_per_node=4 --master_port ${port} train.py \
--data-path /root/userfolder/data/voc2007 \
--dataset voc2007 --ilod "${ilod}" --num-classes 21 \
--base-model models/moco_v2_200ep_pretrain.pth.tar \
--pb body --freeze None \
--epochs 48 --lr-steps 32 44 -b 2 \
--lr-w ${lr_weight} --lr-m ${lr_mask} --optim ${optim} \
--output-dir "${out}" \
--mask-init 1s --mask-scale 1e-2 > $out'.out'