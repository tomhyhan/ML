# import torch

# ratio = torch.tensor([0.5, 1, 2])
# scales = torch.tensor([128, 256, 512])

# h_ratios = ratio.sqrt()
# w_ratios = 1 / h_ratios

# print(h_ratios, w_ratios)

# print(h_ratios * 128)

# print((h_ratios[:, None] * scales[None, :]).view(-1))
# hs = (h_ratios[:, None] * scales[None, :]).view(-1)
# ws = (w_ratios[:, None] * scales[None, :]).view(-1)
# base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

# print(base_anchors)
# base_anchors = base_anchors.round()

# print(torch.arange(0, 64, dtype=torch.int32) * 4)

# s = torch.arange(0, 512, dtype=torch.int32) * 16
# print(base_anchors[0])
# shifts_y, shifts_x = torch.meshgrid(s, s, indexing="ij")

# # print(shifts_y, shifts_x)
# x = shifts_x.reshape(-1)
# y = shifts_y.reshape(-1)
# shifts = torch.stack((x, y, x, y), dim=1)

# print(shifts[0])
# anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
# print(base_anchors.shape)
# print(anchors[0])
