out='test'

python test.py \
--data-path /root/userfolder/data/VOC \
--dataset voc --ilod '[1, 10]' --num-classes 11 \
--base-model model/resnet50-19c8e357.pth \
--pb None --freeze None --pureFasterRCNN \
--epochs 18 --lr-steps 10 15 -b 2 \
--lr-w 1e-3 --lr-m 0 --optim 'SGD' \
--output-dir "${out}" \
--mask-init 1s --mask-scale 1e-2 > "${out}.out"