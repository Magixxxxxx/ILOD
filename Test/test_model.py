import torch
from torchvision.models import resnet
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import sys
sys.path.append(".")

import piggyback_detection

def testDet(model_pth):  
    model = piggyback_detection.fasterrcnn_resnet50_fpn(
        num_classes=11, pretrained=False, base_model="model/fasterrcnn_resnet50_fpn_pretrained.pth",
        mask_init='1s', mask_scale=6e-3, device='cpu'
    )
    state_dict = torch.load(model_pth, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    for name, p in model.named_parameters():
        if 'mask' in name:
            print(name, torch.sum(p.view(-1)>0.005).numpy() / len(p.view(-1)))

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    

if __name__ == '__main__':
    model_pth = "checkpoints/model_19.pth"
    testDet(model_pth)
    
