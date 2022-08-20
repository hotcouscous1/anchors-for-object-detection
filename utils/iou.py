from __init__ import *


def batch_iou(
        bboxes1: Tensor,
        bboxes2: Tensor,
        format: str = 'cxcywh'
) -> Tensor:

    """
    Args:
        bboxes1: (b x m x 4) tensor
        bboxes2: (b x m x 4) tensor
        format: bbox format of both tensors
            'xywh' -> (x_min, y_min, width, height)
            'cxcywh' -> (x_center, y_center, width, height)
            'xyxy' -> (x_min, y_min, x_max, y_max)

    Output:
        (b x m x n) tensor
    """

    if format == 'xywh':
        x_min1 = bboxes1[..., 0]
        y_min1 = bboxes1[..., 1]
        x_max1 = bboxes1[..., 0] + bboxes1[..., 2]
        y_max1 = bboxes1[..., 1] + bboxes1[..., 3]

        x_min2 = bboxes2[..., 0]
        y_min2 = bboxes2[..., 1]
        x_max2 = bboxes2[..., 0] + bboxes2[..., 2]
        y_max2 = bboxes2[..., 1] + bboxes2[..., 3]

    elif format == 'cxcywh':
        x_min1 = bboxes1[..., 0] - bboxes1[..., 2] / 2
        y_min1 = bboxes1[..., 1] - bboxes1[..., 3] / 2
        x_max1 = bboxes1[..., 0] + bboxes1[..., 2] / 2
        y_max1 = bboxes1[..., 1] + bboxes1[..., 3] / 2

        x_min2 = bboxes2[..., 0] - bboxes2[..., 2] / 2
        y_min2 = bboxes2[..., 1] - bboxes2[..., 3] / 2
        x_max2 = bboxes2[..., 0] + bboxes2[..., 2] / 2
        y_max2 = bboxes2[..., 1] + bboxes2[..., 3] / 2

    elif format == 'xyxy':
        x_min1 = bboxes1[..., 0]
        y_min1 = bboxes1[..., 1]
        x_max1 = bboxes1[..., 2]
        y_max1 = bboxes1[..., 3]

        x_min2 = bboxes2[..., 0]
        y_min2 = bboxes2[..., 1]
        x_max2 = bboxes2[..., 2]
        y_max2 = bboxes2[..., 3]

    else:
        raise ValueError("bbox format should be one of 'xywh', 'cxcywh, 'xyxy'")

    x_min1, y_min1, x_max1, y_max1 \
        = x_min1.unsqueeze(2), y_min1.unsqueeze(2), x_max1.unsqueeze(2), y_max1.unsqueeze(2)

    x_min2, y_min2, x_max2, y_max2 \
        = x_min2.unsqueeze(1), y_min2.unsqueeze(1), x_max2.unsqueeze(1), y_max2.unsqueeze(1)

    w_cross = (torch.min(x_max1, x_max2) - torch.max(x_min1, x_min2)).clamp(min=0).to(bboxes1.device)
    h_cross = (torch.min(y_max1, y_max2) - torch.max(y_min1, y_min2)).clamp(min=0).to(bboxes1.device)

    intersect = w_cross * h_cross

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union = area1 + area2 - intersect + 1e-7

    return intersect / union
