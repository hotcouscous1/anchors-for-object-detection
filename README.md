## Anchors for Object Detections

<p align="center">
  <img src="https://github.com/hotcouscous1/Logo/blob/main/TensorBricks_Logo.png" width="500" height="120">
</p>

### Anchor-Maker
Generate anchors to combine with regression predictions of a model. For other cases, check [here](https://github.com/hotcouscous1/Anchors-for-Object-Detection/issues/1).

```python
img_size = 608
anchor_sizes = [32, 64, 128, 256, 512]
anchor_scales = [1, 2**(1/3), 2**(2/3)]
aspect_ratios = [(2**(1/2), 2**(-1/2)), (1, 1), (2**(-1/2), 2**(1/2))]
strides = [8, 16, 32, 64, 128]

anchors = retinanet_anchors(img_size, anchor_sizes, anchor_scales, aspect_ratios, strides)

pred[..., :2] = anchors[..., :2] + (pred[..., :2] * anchors[..., 2:])
pred[..., 2:4] = torch.exp(pred[..., 2:4]) * anchors[..., 2:]
```

### Anchor-Assigner
Assign targets to the anchors, and then return indices of target-assigned anchors and lables of assigned targets.

```python
assigner = retinanet_assigner(0.5, 0.4)
assigns = assigner(labels, anchors)

for pred, assign in zip(preds, assigns):
    fore_idx, fore_label = assign['foreground']
    back_idx, _ = assign['background']

    cls_loss = self.focal_loss(pred[..., 4:], fore_idx, back_idx, fore_label[..., 4:])
    reg_loss = self.smooothL1_loss(pred[..., :4], anchors, fore_idx, fore_label[..., :4])
``` 

## License
BSD 3-Clause License Copyright (c) 2022, hotcouscous1
