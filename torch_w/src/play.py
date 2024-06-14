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

l = torch.nn.Linear(5,3)
print([i for i in l.named_parameters()])