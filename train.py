import torch, torchvision, argparse, os, time, datetime

import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.detection

from torchvision import datasets, transforms
from torch import nn

from utils import utils
from utils import transforms as T
from utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from utils.coco_utils import get_coco
from utils.engine import train_one_epoch, evaluate

import piggyback_detection

def get_dataset(name, image_set, transform, data_path, num_classes):
    paths = {
        "coco": (data_path, get_coco, None), # 修改自定义数据集类别数量：num_classes+1(背景)
        "coco[40,49]": (data_path, get_coco, "[40,49]")
    }
    p, dataset_func, ilod = paths[name]

    dataset = dataset_func(p, image_set=image_set, transforms=transform, ilod=ilod)
    return dataset

def get_detection_model(args):

    model = piggyback_detection.fasterrcnn_resnet50_fpn(
        num_classes=args.num_classes, pretrained=args.pretrained, 
        base_model=args.base_model, mask_init='1s', mask_scale=6e-3,
        device = args.device
        )

    return model
    
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_samplers(args, dataset, dataset_test):
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    return train_batch_sampler, train_sampler, test_sampler    

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-path', default='../data/COCO2017', 
                        help='dataset path')
    parser.add_argument('--dataset', default='coco[40,49]', help='dataset')
    parser.add_argument('--num-classes', default=11, 
                        help='number of classes in dataset(+1 background)', type=int)
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', 
                        help='backbone model of fasterrcnn, options are: \
                        resnet50,vgg16,mobilenet_v2,squeezenet1_0,alexnet,mnasnet0_5')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=18, type=int, metavar='N',
                        help='number of total epochs to run, 30')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[13, 16], nargs='+', type=str, help='decrease lr every step-size epochs,[16, 22]')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='checkpoints', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--pretrained", default=False, action="store_true")

    #piggyback
    parser.add_argument("--base-model", default='', type=str)
    parser.add_argument("--base-classnum", default=91, type=int) #暂时没用

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int)

    return parser.parse_args()

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

    # TODO: Different lr
        
    #

    # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)  
        return

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    args = get_args()
    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)


