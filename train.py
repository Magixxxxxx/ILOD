import torch, torchvision, argparse, os, time, datetime

import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.detection

from torchvision import datasets, transforms
from torch import nn

from utils import utils
from utils import transforms as T
from utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from utils.coco_utils import get_coco,get_voc2007,get_voc0712
from utils.engine import train_one_epoch, train_one_epoch_AdamSGD, evaluate

import piggyback_detection

def get_dataset(name, image_set, data_path, transform, ilod ,num_classes):
    paths = {
        "coco": (data_path, get_coco), # 修改自定义数据集类别数量：num_classes+1(背景)
        "voc2007": (data_path, get_voc2007),
        "voc0712": (data_path, get_voc0712),
    }
    p, dataset_func = paths[name]

    dataset = dataset_func(p, image_set=image_set, transforms=transform, ilod=ilod)
    return dataset

def get_detection_model(args):
    for k,v in vars(args).items(): print(k,v)

    if args.pureFasterRCNN:
        print("pureFasterRCNN")
        from torchvision.models.detection import faster_rcnn
        model = faster_rcnn.fasterrcnn_resnet50_fpn(num_classes=args.num_classes)
        for n,p in model.named_parameters():
            if p.requires_grad:
                print(n)
        return model

    from torchvision.models.detection import faster_rcnn
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models import resnet
    from piggyback_detection import piggyback_resnet

    #1. feature extract
    if 'body' in args.pb:
        print("\npiggyback res50")
        res50 = piggyback_resnet.piggyback_resnet50()
        sd = torch.load(args.base_model, map_location=torch.device('cpu'))
        res50.load_state_dict(sd, strict=False)
    else:
        print("\nbase res50")
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
        res50 = resnet.__dict__['resnet50'](pretrained=True, norm_layer=norm_layer)

    #2. fpn
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:3]
    for name, parameter in res50.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    returned_layers = [1, 2, 3, 4]
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = res50.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    
    if 'fpn' in args.pb:
        print("piggyback fpn")
        backbone = piggyback_detection.backbone_utils.BackboneWithFPN(res50, return_layers, in_channels_list, out_channels)

        backbone_dict = {}
        for k,v in torch.load(args.base_model, map_location=torch.device('cpu')).items(): 
            if 'backbone' in k: 
                backbone_dict[k[9:]] = v
        backbone.load_state_dict(backbone_dict, strict=False)
    else:
        print("base fpn")
        backbone = BackboneWithFPN(res50, return_layers, in_channels_list, out_channels)

    #3. detector
    print("base detector")
    model = faster_rcnn.FasterRCNN(backbone,num_classes=args.num_classes)

    print("\nParameters Requires grad: ")
    for n,p in model.named_parameters():
        for freeze in args.freeze:
            if freeze in n: 
                p.requires_grad_(False)
        if p.requires_grad:
            print(n)

    return model
    
def get_transform(train):
    trans = []
    trans.append(T.ToTensor())
    # trans.append(T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)))
    # FasterRCNN已经默认加了这个， 包括resize
    if train:
        trans.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(trans)

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

def get_optimizer(args, model):
    # distributed之后，参数需加上module / pb mode 0,1
    masks = [p for n, p in model.module.named_parameters() if 'mask' in n]
    nomask_params = [p for n, p in model.module.named_parameters() if 'mask' not in n]
    params = [p for n, p in model.module.named_parameters() if p.requires_grad]

    if args.optim == 'Adam':
        if args.lr_m:
            print('\nAdam lr m:{} w:{}'.format(args.lr_m,args.lr_w))
            optimizer = torch.optim.Adam([
                {'params': masks, 'lr': args.lr_m, 'weight_decay':args.weight_decay},
                {'params': nomask_params, 'lr': args.lr_w, 'weight_decay':args.weight_decay}
            ])
        else:
            print('\nAdam lr {}'.format(args.lr_w))
            optimizer = torch.optim.Adam(params, lr=args.lr_w, weight_decay=args.weight_decay)

    elif args.optim == 'SGD':
        if args.lr_m:
            print('\nSGD lr m:{} w:{}'.format(args.lr_m,args.lr_w))
            optimizer = torch.optim.SGD([
                {'params':masks, 'lr':args.lr_m, 'momentum':args.momentum, 'weight_decay':args.weight_decay},
                {'params':nomask_params, 'lr': args.lr_w, 'momentum':args.momentum, 'weight_decay':args.weight_decay},
            ])
        else:
            print('\nSGD lr {}'.format(args.lr_w))
            optimizer = torch.optim.SGD(params, lr=args.lr_w, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer 

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='/home/zhaojiawei/Data/COCO2017', 
                        help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--ilod', default='[47,48]')
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
    parser.add_argument('--lr-m', type=float,
                        help='0.02 default for 8 gpus and 2 images_per_gpu')
    parser.add_argument('--lr-w', default=1e-4, type=float)       
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[13, 16], nargs='+', type=int, help='decrease lr every step-size epochs,[16, 22]')
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
    parser.add_argument('--pb', default=[], nargs='*', type=str,
                        help="piggyback mode :body, fpn, rpn, roi")
    parser.add_argument("--freeze", default=[], nargs='*', type=str,
                        help="freeze params :body, fpn, rpn, roi")
    parser.add_argument("--optim", default='Adam', type=str)
    parser.add_argument("--pureFasterRCNN", default=False, action='store_true')

    parser.add_argument("--mask-init", default='1s', type=str)
    parser.add_argument("--mask-scale", default=1e-2, type=float)

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int)

    return parser.parse_args()

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    print("\nLoading data")
    dataset = get_dataset(args.dataset, "trainval", args.data_path, 
                        get_transform(train=True), args.ilod, args.num_classes)
    dataset_test= get_dataset(args.dataset, "test", args.data_path,
                        get_transform(train=False), args.ilod, args.num_classes)

    print("\nCreating data loaders")
    train_batch_sampler, train_sampler, test_sampler = get_samplers(args, dataset, dataset_test)

    data_loader = torch.utils.data.DataLoader(dataset, 
        batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("\nCreating model")

    model = get_detection_model(args)
    model.to(device)
    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.optim == 'AdamSGD':
        print('AdamSGD lr Adam:{} SGD:{}'.format(args.lr_m, args.lr_w))
        optimizerMask = torch.optim.Adam([p for n, p in model.module.named_parameters() if 'mask' in n], 
            lr=args.lr_m, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD([p for n, p in model.module.named_parameters() if 'mask' not in n], 
            lr=args.lr_w, momentum=args.momentum, weight_decay=args.weight_decay)

        lr_schedulerMask  = torch.optim.lr_scheduler.MultiStepLR(optimizerMask, milestones=args.lr_steps, gamma=args.lr_gamma)
        lr_scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    else:
        optimizer = get_optimizer(args, model)
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

    print("\nStart training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.optim == 'AdamSGD':
            train_one_epoch_AdamSGD(model, optimizerMask, optimizer, data_loader, device, epoch, args.print_freq)
            lr_schedulerMask.step()
            lr_scheduler.step()
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizerMask': optimizerMask.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'lr_schedulerMask': lr_schedulerMask.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        else:
            train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
            lr_scheduler.step()
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
