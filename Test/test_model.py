import torch
from torchvision.models import resnet
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import sys,os
sys.path.append(".")

import piggyback_detection

def testMask(path):
    from collections import defaultdict
    mask_density = defaultdict(list)
    mask_mean = defaultdict(list)

    if path.endswith('.pth'):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        for name,p in state_dict['model'].items():
            if 'mask' in name:
                density = torch.sum(p.view(-1)<0.005).numpy() / len(p.view(-1))
                mean = torch.mean(p.view(-1))
                max_mask = torch.max(p.view(-1))
                min_mask = torch.min(p.view(-1))
                print("{} \n zeroed:{:.3f} mean:{:.3f} max:{:.3f} min:{:.3f}".format(name,density,mean,max_mask,min_mask))
    else:
        for model_pth in os.listdir(path):
            state_dict = torch.load(os.path.join(path,model_pth), map_location=torch.device('cpu'))
            for name,p in state_dict['model'].items():
                if 'mask' in name:
                    density = torch.sum(p.view(-1)>0.005).numpy() / len(p.view(-1))
                    mean = torch.mean(p.view(-1))
                    mask_density[name].append(density)
                    mask_mean[name].append(mean)

    # np.save('mask_density.npy',mask_density)
    # np.save('mask_mean.npy',mask_mean)

def analyseParams(model):
    layer = 'backbone.body.layer3.3.conv1.mask_real'
    state_dict = torch.load(model, map_location=torch.device('cpu'))
    for name,p in state_dict['model'].items():
        if layer in name:

            print(p.view(-1))
    print(state_dict['lr_scheduler'])
if __name__ == '__main__':
    model = "voc[16, 20]pb[body]_Adam1e-4 5*/model_11.pth"
    path = "checkpoints/"
    analyseParams(model)
    

