import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, emp_dim, feedforward_dim, device="cpu", dtype=torch.float32):
        super().__init__()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(emp_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, emp_dim) 
        )
        
    def forward(self, x):
        return self.feed_forward(x)           