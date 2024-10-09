import torch 
from torch import nn
num_heads = 1
window_size = [2,2]

def define_relative_position_bias_table():
    # define a parameter table of relative position bias
    relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
    )  # 2*Wh-1 * 2*Ww-1, nH
    nn.init.trunc_normal_(relative_position_bias_table, std=0.02)
    # relative_position_bias_table = torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        
    return relative_position_bias_table

def define_relative_position_index():
    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
    return relative_position_index

relative_position_bias_table = define_relative_position_bias_table()

relative_position_index = define_relative_position_index()

print(relative_position_bias_table.shape)
print(relative_position_index.shape)

print(relative_position_bias_table[relative_position_index].shape)