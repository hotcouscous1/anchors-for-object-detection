import torch.nn as nn
from utils.anchor_priors import *


class Anchor_Maker(nn.Module):

    __doc__ = r"""
        Anchors are generated to be mapped to the predictions of all detection models.

        Args:
            anchor_priors: relative sizes to a single grid cell of a feature map, regardless of its stride.
            strides: strides of each level on which the anchors are placed, to the input image.
            center: to place anchors on the center of each grid.
                if 'center' is False, anchors are placed on the left-top, which is the case of Yolo.
            clamp: to bound all anchor-values between 0 and image size.
            relative: to normalize all anchor-values by image size.

        Output:
            1d tensor ordered in (stride * height * width * num_anchors per grid_cell)
        """

    def __init__(self,
                 anchor_priors: Tensor or List[Tensor],
                 strides: List[int],
                 center: bool = True,
                 clamp: bool = False,
                 relative: bool = False
                 ):

        super().__init__()

        if type(anchor_priors) is Tensor or len(anchor_priors) != len(strides):
            anchor_priors = len(strides) * [anchor_priors]

        self.priors = anchor_priors
        self.strides = strides
        self.center = center
        self.clamp = clamp
        self.relative = relative


    def forward(self, img_size: int) -> Tensor:
        all_anchors = []

        for stride, priors in zip(self.strides, self.priors):
            stride_anchors = []

            num_grid = math.ceil(img_size / stride)

            if self.center:
                grid = torch.arange(num_grid, device=device).repeat(num_grid, 1).float() + 0.5
            else:
                grid = torch.arange(num_grid, device=device).repeat(num_grid, 1).float()

            x = grid * stride
            y = grid.t() * stride

            boxes = (stride * priors)

            for box in boxes:
                w = torch.full([num_grid, num_grid], box[0], device=device)
                h = torch.full([num_grid, num_grid], box[1], device=device)
                anchor = torch.stack((x, y, w, h))

                stride_anchors.append(anchor)

            stride_anchors = torch.cat(stride_anchors).unsqueeze(0)
            stride_anchors = stride_anchors.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

            all_anchors.append(stride_anchors)
        all_anchors = torch.cat(all_anchors, dim=1)

        if self.clamp:
            all_anchors = torch.clamp(all_anchors, 0, img_size)

        if self.relative:
            all_anchors /= img_size

        return all_anchors




def yolo_anchors(
        img_size: int,
        anchor_sizes: List[List],
        strides: list
) -> List[Tensor]:

    """
    Examples:
        img_size = 416
        anchor_sizes = [[(10, 13), (16, 30), (33, 23)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(116, 90), (156, 198), (373, 326)]]

        strides = [8, 16, 32]
        anchors = yolo_anchors(img_size, anchor_sizes, strides)
    """

    anchors = []
    anchor_priors = yolo_anchor_priors(anchor_sizes, strides)

    for i, s in enumerate(strides):
        s_anchors = [Anchor_Maker(p.unsqueeze(0), [s], False, False, False).to(device)(img_size)
                     for p in anchor_priors[i]]
        s_anchors = torch.cat(s_anchors, 1)

        anchors.append(s_anchors)
    return anchors



def retinanet_anchors(
        img_size: int,
        anchor_sizes: list,
        anchor_scales: list,
        aspect_ratios: List[tuple],
        strides: list
) -> Tensor:

    """
    Examples:
        img_size = 608
        anchor_sizes = [32, 64, 128, 256, 512]
        anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
        aspect_ratios = [(2 ** (1 / 2), 2 ** (-1 / 2)), (1, 1), (2 ** (-1 / 2), 2 ** (1 / 2))]
        strides = [8, 16, 32, 64, 128]

        anchors = retinanet_anchors(img_size, anchor_sizes, anchor_scales, aspect_ratios, strides)
    """

    anchor_priors = retinanet_anchor_priors(anchor_sizes, anchor_scales, aspect_ratios, strides)
    anchors = Anchor_Maker(anchor_priors, strides, True, False, False).to(device)(img_size)
    return anchors



def ssd_anchors(
        img_size: int,
        anchor_sizes: list,
        upper_sizes: list,
        strides: list,
        num_anchors: list
) -> Tensor:

    """
    Examples:
        img_size = 300
        anchor_sizes = [21, 45, 99, 153, 207, 261]
        upper_sizes = [45, 99, 153, 207, 261, 315]
        strides = [8, 16, 32, 64, 100, 300]
        num_anchors = [4, 6, 6, 6, 4, 4]

        anchors = ssd_anchors(img_size, anchor_sizes, upper_sizes, strides, num_anchors)
    """

    anchor_priors = ssd_anchor_priors(anchor_sizes, upper_sizes, strides, num_anchors)
    anchors = Anchor_Maker(anchor_priors, strides, True, True, True).to(device)(img_size)
    return anchors

