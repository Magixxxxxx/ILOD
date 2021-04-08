from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter

from .generalized_rcnn import GeneralizedRCNN
from .rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from .roi_heads import RoIHeads
from .transform import GeneralizedRCNNTransform
from .backbone_utils import piggyback_resnet_fpn_backbone

from .layers import ElementWiseConv2d, ElementWiseLinear
from .rpn import Piggyback_PRNHead

__all__ = [
    "FasterRCNN", "fasterrcnn_resnet50_fpn",
]


class FasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # piggyback parameters
                 incremental=None,
                 mask_init='1s', 
                 mask_scale=6e-3):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = Piggyback_PRNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0],
                mask_init=mask_init, mask_scale=mask_scale)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = Piggback_TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size, 
                mask_init=mask_init, mask_scale=mask_scale)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = Piggback_FastRCNNPredictor(
                representation_size, num_classes,
                incremental, mask_init=mask_init, mask_scale=mask_scale)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class Piggback_TwoMLPHead(nn.Module):

    def __init__(self, in_channels, representation_size,
        mask_init='1s', mask_scale=6e-3):
        super(Piggback_TwoMLPHead, self).__init__() 

        self.fc6 = ElementWiseLinear(in_channels, representation_size, 
            mask_init=mask_init, mask_scale=mask_scale)
        self.fc7 = ElementWiseLinear(representation_size, representation_size, mask_init=mask_init, mask_scale=mask_scale)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))

        x = F.relu(self.fc7(x))

        return x


class Piggback_FastRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_classes,
        incremental, mask_init='1s', mask_scale=6e-3):
        super(Piggback_FastRCNNPredictor, self).__init__()
        #incre_cls_score：真正的梯度传导对象
        self.cls_score = nn.Linear(in_channels, num_classes) 
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

        #新分类，预测头
        self.incre_cls_score = nn.Linear(in_channels, incremental)
        self.incre_bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.incre_cls_score(x)
        bbox_deltas = self.incre_bbox_pred(x)

        return scores, bbox_deltas


def fasterrcnn_resnet50_fpn(
    progress=True, num_classes=11, base_model=None, 
    mask_init='1s', mask_scale=1e-2, device="cpu",
    **kwargs):

    #---------piggyback---------#
    if base_model:

        backbone = piggyback_resnet_fpn_backbone(mask_init, mask_scale)
        state_dict = torch.load(base_model, map_location=torch.device(device))

        #基类别数目，仅接收参数/增量类别数目，新fc头
        base_num_classes = len(state_dict['roi_heads.box_predictor.cls_score.weight'])
        print("get piggyback base:{} incre:{}".format(base_num_classes, num_classes))

        model = FasterRCNN(backbone, base_num_classes, incremental=num_classes)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("get fasterrcnn cls:{}".format(num_classes))
        model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes,
                                        pretrained=True)
    #---------piggyback---------#

    return model
