from torch import nn

class WindowPartition(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x, p):
        B, C, H, W = x.shape
        
        x = x.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0,2,4,3,5,1)
        x = x.reshape(B, (H//p) * (W//p), p*p, C)
    
        return x
    
class WindowDepartition(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x, p, hp, wp):
        B, HpWp, pp, C = x.shape

        x = x.reshape(B, hp, wp, p, p, C)
        x = x.permute(0,5,1,3,2,4)
        x = x.reshape(B, C, hp*p, wp*p)
    
        return x
    