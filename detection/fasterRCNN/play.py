import torch

image = torch.rand(2, 3, 1200, 600)

scale = 1000/1200

image = torch.nn.functional.interpolate(
    image,
    size=None,
    scale_factor=scale,
    recompute_scale_factor=True,
    mode="bilinear",
    align_corners=False
)

print(image.shape)
