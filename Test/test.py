import torch
from torchvision.models import resnet

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from ..piggyback_detection.piggyback_resnet import resnet50

def testModel():
    model = resnet50()
    for name,param in model.named_parameters():
        print(name)
  

if __name__ == '__main__':
    testModel()
    
