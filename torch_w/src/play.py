import torch

# class testing:
#     def __init__(self, **kwargs):
#         print("hello")
#         print(**kwargs)
#         super().__setattr__('training', True)
#         super().__init__(**kwargs)
#         print(**kwargs)
        
        
# t = testing()
# print(t.training)
X = torch.randn(10,3,32,32)
conv = torch.nn.Conv2d(3,128, 1)
# print([i for i in l.named_parameters()])
# print(conv.weight.shape)
outx = conv(X)
# print(outx.shape)
conv2 = torch.nn.Conv2d(128,128, 3, groups=32, stride=2, padding=1)
print("w:", conv2.weight.shape)
out_x = conv2(outx)
print(out_x.shape)