from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import torch
data = COCO("/root/userfolder/data/VOC/annotations/voc_val2007.json")
cocoGt = data.getAnnIds()
E = COCOeval(data)

from torchvision.models.detection import faster_rcnn
model = faster_rcnn.fasterrcnn_resnet50_fpn(num_classes=21)
sd = torch.load("voc[1, 20]_SGD5e-3b2*4/model_24.pth", map_location=torch.device('cpu'))['model']

model.load_state_dict(sd)
outputs = model(images)

outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

E.update()
E.accumulate()
print(E.evaluate())