import torch 
from torch import nn

class ShiftedWindowAttn(nn.Module):
    
    def __init__(
        self,
        emb_dim,
        window_size,
        shift_size,
        num_heads,
        dropout,
        qkv_bias=True,
        proj_bias=True
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(emb_dim, emb_dim*3, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=proj_bias)
        
        relative_position_table = self.get_relative_position_table()
        relative_position_index = self.get_relative_position_index()
        
        num_windows = window_size[0] * window_size[1]
        self.relative_position = relative_position_table[relative_position_index] \
            .view(num_windows, num_windows, -1) \
            .permute(2,0,1) \
            .unsqueeze(0) # for batch
        
    def get_relative_position_table(self):
        relative_position_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[0] - 1), self.num_heads)
        )
        nn.init.trunc_normal_(relative_position_table, std=0.02)
        return relative_position_table
    
    def get_relative_position_index(self):
        H = torch.arange(self.window_size[0])
        W = torch.arange(self.window_size[1])
        coords = torch.meshgrid(H, W, indexing="ij")
        coords_stack = torch.stack(coords)
        coords_flatten = torch.flatten(coords_stack, dims=1)
        coords_flatten = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        coords_perm = coords_perm.permute(1,2,0)
        
        coords_perm[:,:,0] += self.window_size[0] - 1
        coords_perm[:,:,1] += self.window_size[0] - 1
        coords_perm[:,:,0] *= 2*self.window_size[0] - 1

        coords_index = coords_perm.sum(-1).flatten()
        return coords_index
    
    def forward(self, x):
        return shifted_window(
            x,
            self.emb_dim,
            self.window_size,
            self.sh
            
        )

def shifted_window(x):
    pass


