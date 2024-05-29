import torch

def kaiming_initialization(intput_dim):
    """
        Xavier weight initialization for Linear Layer
        kaiming weight initialization for Conv layer. 
        
        Formula: 
            Var_in == var_out
            
        Inputs:
            input_dim: (C,H,W) input dimension
        Outpus:
            w: (C,H,W) kaiming initialized weights  
    """
    w = torch.random()