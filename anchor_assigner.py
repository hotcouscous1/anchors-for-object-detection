from __init__ import *
import torch.nn as nn
from utils.iou import batch_iou


class Anchor_Assigner(nn.Module):

    __doc__ = r"""
        This module assigns target labels to the anchors, which is necessary in loss functions
        for anchor-based detection models.

        Args:
            fore_th: threshold to filter anchors to foreground
            back_th: threshold to filter anchors to background
            max_for_target: regardless of the thresholds, 
            whether to filter anchors with the largest IoU for given targets to foreground, or not
            foregrond_only: whether not to use backgrounds, or not
            bbox_format: bbox format for IoUs between label and anchors

            labels: shape of (batch size, num_targets, 4 + num_classes)
            anchors: shape of (1, num_pred, 4)

        Output:
            [{'foreground': [indices of target-assinged anchors, labels of targets assigned to the anchors],
              'background': [indices of background anchors, labels of background assigned to the anchor]},
             {'foreground': ...,
              'background': ...},
             ...

        * remainders, which are not foregrounds nor backgrounds, are invalid targets
    """

    def __init__(self,
                 fore_th: float,
                 back_th: float = None,
                 max_for_target: bool = False,
                 foreground_only: bool = True,
                 bbox_format: str = 'cxcywh'):

        super().__init__()

        self.fore_th = fore_th
        self.back_th = back_th
        self.max_for_target = max_for_target
        self.foreground_only = foreground_only
        self.bbox_format = bbox_format


    def forward(self, labels: Tensor, anchors: Tensor) -> List[dict]:
        ious = batch_iou(anchors, labels[..., :4], self.bbox_format)
        batch_assign = []

        for i, label in enumerate(labels):

            if not (self.fore_th or self.max_for_target):
                raise ValueError("one of them must be given")

            if not self.fore_th:
                self.fore_th = 1.0 + 1e-5

            max_iou_anchor, target_for_anchor = torch.max(ious[i], dim=1)
            fore_mask = max_iou_anchor >= self.fore_th

            if self.max_for_target:
                max_iou_target, anchor_for_target = torch.max(ious[i], dim=0)
                fore_mask_target = torch.zeros(fore_mask.size(), device=device).bool()
                fore_mask_target[anchor_for_target] = True

                fore_mask = torch.logical_or(fore_mask, fore_mask_target)


            back_mask = torch.logical_not(fore_mask)

            if self.back_th:
                back_mask = torch.logical_and(back_mask, max_iou_anchor < self.back_th)
                # remainders, which are not foregrounds nor backgrounds, are invalid targets

            assigned_target = label[target_for_anchor]

            if self.foreground_only:
                batch_assign.append({'foreground': [fore_mask.nonzero(as_tuple=True)[0], assigned_target[fore_mask]]})
            else:
                batch_assign.append({'foreground': [fore_mask.nonzero(as_tuple=True)[0], assigned_target[fore_mask]],
                                     'background': [back_mask.nonzero(as_tuple=True)[0], assigned_target[back_mask]]})

        return batch_assign




# bbox_format depends on either model predictions or loss functions.

def yolo_assigner(
        fore_th: float = None,
        back_th: float = 0.5
) -> Anchor_Assigner:

    return Anchor_Assigner(fore_th, back_th, max_for_target=True, foreground_only=False, bbox_format='cxcywh').to(device)


def retinanet_assigner(
        fore_th: float = 0.5,
        back_th: float = 0.4
) -> Anchor_Assigner:

    return Anchor_Assigner(fore_th, back_th, max_for_target=False, foreground_only=False, bbox_format='cxcywh').to(device)


def ssd_assigner(
        fore_th: float = 0.5,
        back_th: float = 0.5
) -> Anchor_Assigner:

    return Anchor_Assigner(fore_th, back_th, max_for_target=True, foreground_only=False, bbox_format='cxcywh').to(device)


def faster_rcnn_assigner(
        fore_th: float = 0.7,
        back_th: float = 0.3
) -> Anchor_Assigner:

    return Anchor_Assigner(fore_th, back_th, max_for_target=True, foreground_only=False, bbox_format='cxcywh').to(device)


