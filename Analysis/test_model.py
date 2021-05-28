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

def analyseParams(model, layer):
    state_dict = torch.load(model, map_location=torch.device('cpu'))
    for name,p in state_dict['model'].items():
        if layer in name:
            print(p.view(-1))

def compareModels(model1, model2):
    state_dict1 = torch.load(model1, map_location=torch.device('cpu'))
    state_dict2 = torch.load(model2, map_location=torch.device('cpu'))
    for (name1,p1),(name2,p2) in zip(state_dict1['model'].items(),state_dict2['model'].items()):
        if 'mask' in name1:
            active1 = p1.view(-1)>0.005
            active2 = p2.view(-1)>0.005
            common = [a and b for a,b in zip(active1, active2)]
            com_by_all = sum(common).numpy() / len(p1.view(-1))
            com_by_1s = sum(common).numpy() / sum(active1).numpy()
            print("%s %.2f %.2f" % (name1, com_by_1s, com_by_all))
            # print(sum(p1.view(-1)>0.005==p2.view(-1)>0.005), len(p1.view(-1)))

def compareLayers(model):
    state_dict = torch.load(model, map_location=torch.device('cpu'))
    for name,p in state_dict['model'].items():
        if 'backbone.body.conv1.mask_real' == name:
            pre_active = p.view(-1)>0.005
            continue
        if 'mask' in name:
            pre_active = active
            active = p.view(-1)>0.005
            print(name, sum(active == pre_active).numpy()/len(p.view(-1)))
            # print(sum(p1.view(-1)>0.005==p.view(-1)>0.005), len(p1.view(-1)))


if __name__ == '__main__':

    model1 = "voc[16, 20]_pb[body]moco200_Adam:1e-4_1e-5/model_47.pth"
    model2 = "voc[16, 20]_pb[body]moco2002_Adam:1e-4_1e-5/model_47.pth"
    # path = "checkpoints/"
    layer = "backbone.body.layer4.2.conv3.weight"
    
    compareModels(model1, model2)

