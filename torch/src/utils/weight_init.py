import torch

def kaiming_initialization(D_in, D_out, k=None, relu=True, device="cpu", dtype=torch.float):
    """
        Xavier weight initialization for Linear Layer
        kaiming weight initialization for relu. 
        
        Formula: 
            Var_in == var_out / sqrt((1 or 2) / D_in)
            
        Inputs:
            input_dim: (C,H,W) input dimension
        Outpus:
            w: (C_out, C_in, H, W) or (C_in, C_out) kaiming initialized weights  
    """
    gain = 2 if relu else 1
    
    if k == None:
        w = torch.randn(D_in, D_out, device=device, dtype=dtype) * (gain / D_in)**1/2
    else:
        w = torch.randn(D_out, D_in , k, k, device=device, dtype=dtype) * (gain / D_in)**1/2
    
    return w
