import torch

def compute_numeric_gradients(f, x: torch.Tensor, dout=None, h=1e-7):
    """
        Compute numeric gradients using center difference
    
        diff = (f(x + h) - f(x - h)) / h 
    """
    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    if dout is None:
        dx = f(x)
        dout = torch.ones_like(dx)
    dout = dout.flatten()
    
    for i in range(flat_x.shape[0]):
        old_val = flat_x[i].item()
        
        flat_x[i] = old_val + h 
        fpx = f(x).flatten()
        flat_x[i] = old_val - h 
        fmx = f(x).flatten()
        flat_x[i] = old_val
        
        diff = (fpx - fmx) /  (2 * h)
        flat_grad[i] = torch.dot(diff, dout).item()

    return grad    

def rel_error(dx, dx_num, eps=1e-10):
    top = (dx - dx_num).abs().max().item()
    bot = (dx.abs() + dx_num.abs()).clamp(min=eps).max().item()
    return top / bot
