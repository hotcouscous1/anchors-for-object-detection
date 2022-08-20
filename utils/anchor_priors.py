from __init__ import *


def yolo_anchor_priors(
        anchor_sizes: List[List],
        strides: list
) -> List[Tensor]:

    anchor_priors = []
    for stride, sizes in zip(strides, anchor_sizes):
        stride_priors = [(w / stride, h / stride) for w, h in sizes]
        anchor_priors.append(torch.Tensor(stride_priors))

    return anchor_priors



def retinanet_anchor_priors(
        anchor_sizes: list,
        anchor_scales: list,
        aspect_ratios: List[tuple],
        strides: list
) -> List[Tensor]:

    anchor_priors = []
    for stride, size in zip(strides, anchor_sizes):
        stride_priors = [[(size / stride) * s * r[0], (size / stride) * s * r[1]]
                         for s in anchor_scales
                         for r in aspect_ratios]

        anchor_priors.append(torch.Tensor(stride_priors))

    return anchor_priors



def ssd_anchor_priors(
        anchor_sizes: list,
        upper_sizes: list,
        strides: list,
        num_anchors: list
) -> List[Tensor]:

    anchor_scales = [size / stride for size, stride in zip(anchor_sizes, strides)]
    upper_scales = [upper / stride for upper, stride in zip(upper_sizes, strides)]

    anchor_priors = []
    for i in range(len(strides)):
        scale, upper = anchor_scales[i], upper_scales[i]
        aspect_ratios = [[1, 1], [math.sqrt(scale * upper) / scale] * 2]

        if num_anchors[i] == 4:
            ratios = [2]

        elif num_anchors[i] == 6:
            ratios = [2, 3]

        elif num_anchors[i] == 8:
            ratios = [1.60, 2, 3]

        else:
            raise ValueError('make num_anchors == 4 or 6 or 8')

        for r in ratios:
            aspect_ratios += [[math.sqrt(r), 1 / math.sqrt(r)], [1 / math.sqrt(r), math.sqrt(r)]]

        stride_priors = [[scale * a[0], scale * a[1]] for a in aspect_ratios]

        stride_priors = torch.Tensor(stride_priors)
        anchor_priors.append(stride_priors)

    return anchor_priors
