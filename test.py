import torch, torchvision, argparse, os, time, datetime
import numpy as np
from torch import nn

from utils import utils
from utils import transforms as T
from utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from utils.coco_utils import get_coco
from utils.engine import train_one_epoch, evaluate

import piggyback_detection

from train import get_dataset, get_detection_model, get_transform,get_transform, get_samplers, get_args, get_detection_model_NoFPN

def main(args):
    print(args)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    print("Loading data")
    dataset = get_dataset(args.dataset, "trainval", 
                        get_transform(train=True), args.data_path, args.num_classes)
    dataset_test= get_dataset(args.dataset, "test", 
                        get_transform(train=False), args.data_path,args.num_classes)

    print("Creating data loaders")
    train_batch_sampler, train_sampler, test_sampler = get_samplers(args, dataset, dataset_test)

    data_loader = torch.utils.data.DataLoader(dataset, 
        batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = get_detection_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]


def test(args):
    args.pb = ['body']
    args.device = 'cpu'
    args.base_model = "model/fasterrcnn_resnet50_fpn_pretrained.pth"
    model = get_detection_model(args)


def check_parameters(net):
    '''
        Returns module parameters. Mb
        backbone.body 23.51 pb +46.96(0.73)     
        backbone.fpn 3.34 pb +3.34(0.10) 
        rpn 0.59 
        roi_heads 13.95W
    '''
    parameters = sum(param.numel() for name,param in net.named_parameters() if 'box_predict' in name)
    return parameters / 10**6

def test2(args):
    from torchvision.models import resnet50
    backbone = resnet50()
    for n in backbone.modules():
        print(n)

if __name__ == "__main__":
    args = get_args()
    if args.output_dir:
        utils.mkdir(args.output_dir)
    test(args)


