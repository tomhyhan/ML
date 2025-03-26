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

import torch

# x = torch.rand(2, 3)
# print(x)

# _, t = x.max(dim=0)
# b, _ = x.max(dim=1)

# print(b)

# print(torch.where(x == b[:, None]))
# r = torch.where(x == b[:, None])[1]
# print("t", t)
# print(t[r])
# best_match_gt_idx -> (num_anchors_in_image)

# best_match_iou = [0.8, 0.7, 0.8]  # Highest IoU for each anchor
# best_match_gt_idx = [0, 1, 0]

# iou_matrix = torch.tensor([[0.8, 0.2, 0.8],  # gt_box_0
#                            [0.3, 0.7, 0.4]])  # gt_box_1
# best_anchor_iou_for_gt = torch.tensor([0.8, 0.7])
# print("max", iou_matrix.max(dim=0)[1])
# gt_pred_pair_with_highest_iou = torch.where(
#     iou_matrix == best_anchor_iou_for_gt[:, None])
# # gt_pred_pair_with_highest_iou = ([0, 0, 1], [0, 2, 1])
# print(iou_matrix == best_anchor_iou_for_gt[:, None])
# print(best_anchor_iou_for_gt[:, None])
# print(gt_pred_pair_with_highest_iou)

# print(iou_matrix.max(dim=0)[1][gt_pred_pair_with_highest_iou[1]])
# pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
# best_match_gt_idx = iou_matrix.max(dim=0)[1]

# print("b", best_match_gt_idx)
# print("g", pred_inds_to_update)
# print("bg", best_match_gt_idx[pred_inds_to_update])
# # b[g] = g
# best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx[pred_inds_to_update]
# # best_match_gt_idx= best_match_gt_idx[best_match_gt_idx < 0.3]

# a = torch.tensor([1,2,3])
# b = torch.tensor([0,1])

# c = torch.tensor([10,11,13])
# a[b] = c[b]

# print(a)

x = torch.tensor([0,True,True,0])

print( torch.where(x ))