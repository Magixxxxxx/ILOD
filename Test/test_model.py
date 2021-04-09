import torch
from torchvision.models import resnet
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import sys
sys.path.append(".")

import piggyback_detection

def testDet(model_pth):  

    state_dict = torch.load(model_pth, map_location=torch.device('cpu'))

    for name,p in state_dict['model'].items():
        if 'mask' in name:
            print(name, torch.sum(p.view(-1)>0.005).numpy() / len(p.view(-1)))

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    

if __name__ == '__main__':
    model_pth = "checkpoints/model_1.pth"
    testDet(model_pth)
    

