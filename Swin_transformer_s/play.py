import torch

window_size = [4,4]
H = torch.arange(window_size[0])
W = torch.arange(window_size[1])
coords = torch.meshgrid(H, W, indexing="ij")
coords_stack = torch.stack(coords)
coords_flatten = coords_stack.flatten(1)
coords_flatten = coords_flatten[:,:,None] - coords_flatten[:,None,:]
coords_perm = coords_flatten.permute(1,2,0)

coords_perm[:,:,0] += window_size[0] - 1
coords_perm[:,:,1] += window_size[1] - 1
coords_perm[:,:,0] *= 2*window_size[1] - 1
coords_index = coords_perm.sum(-1).flatten()
print(coords_index.shape)
print(coords_index)

table = torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), 1)

print(table[coords_index].shape)
