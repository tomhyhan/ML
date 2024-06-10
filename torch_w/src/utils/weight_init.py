import torch

def kaiming_init(D_out, D_in, k=None, relu=True, device="cpu", dtype=torch.float32):
    
    gain = 2 if relu else 1
    
    if k == None:
        weights = torch.randn(D_in, D_out, device=device, dtype=dtype) * (gain / D_in)**0.5    
    else:
        weights = torch.randn(D_out, D_in, k, k, device=device, dtype=dtype) * (gain / (D_in*k*k))**0.5
        
    return weights    
    
