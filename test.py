import torch, torchvision, argparse, os, time, datetime
import numpy as np
from torch import nn

from utils import utils
from utils import transforms as T
from utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from utils.coco_utils import get_coco
from utils.engine import train_one_epoch, evaluate

import piggyback_detection

from train import get_dataset,get_detection_model,get_transform,get_transform, get_samplers,get_args

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
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     args.start_epoch = checkpoint['epoch'] + 1

    # if args.test_only:
    #     evaluate(model, data_loader_test, device=device)  
    #     return

    # print("Start training")
    # start_time = time.time()

    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #     train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
    #     lr_scheduler.step()
    #     if args.output_dir:
    #         utils.save_on_master({
    #             'model': model_without_ddp.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'lr_scheduler': lr_scheduler.state_dict(),
    #             'args': args,
    #             'epoch': epoch},
    #             os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
    #     evaluate(model, data_loader_test, device=device)

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))

def test(args):
    args.device = 'cpu'
    # for n,p in sd.items():
    #     print(n)
    model = get_detection_model(args)
    dataset = get_dataset(args.dataset, "trainval", args.data_path, 
                        get_transform(train=True), args.ilod, args.num_classes)
    dataset_test= get_dataset(args.dataset, "test", args.data_path,
                        get_transform(train=False), args.ilod, args.num_classes)
    
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


