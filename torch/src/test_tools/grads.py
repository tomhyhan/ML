import torch 
import random
        
def grad_check_sparse(f, x, dout, analytic_grads, num_checks=10, h=1e-7):
    """
        computes numeric gradient of f at x using h (finite difference).
        
        Inputs:
            f: forward pass function
            x: torch tensor to compute gradient
            dout: upstream gradient
            h: finite differences
        Outputs:
            grad: the result of x gradient by finite approximation
    """
    x_flat = x.contiguous().flatten()
    eps = 1e-12

    if dout is None:
        z = f(x)
        dout = torch.ones_like(z)

    upstream_grad = dout.flatten()
    analytic_grads = analytic_grads.flatten()
    
    error = 0
    for _ in range(num_checks):
        ix = tuple([random.randrange(m) for m in x_flat.shape])

        old_val = x_flat[ix].item()
        x_flat[ix] = old_val + h
        fph = f(x)
        x_flat[ix] = old_val - h
        fmh = f(x)
        x_flat[ix] = old_val

        local_grad = (fph.flatten() - fmh.flatten()) / (2 * h)
        num_grad = torch.dot(upstream_grad, local_grad).item()
        analytic_grad = analytic_grads[ix].item()
        
        top = abs(num_grad - analytic_grad)
        bot = (abs(num_grad) + abs(analytic_grad)) + eps        
        ix_error = top / bot 

        msg = "numerical: %f analytic: %f, relative error: %e"
        print(msg % (num_grad, analytic_grad, ix_error))
        error += ix_error
        
    print(f"Average error on {num_checks} number of checks: {error}")

def compute_numeric_gradient(f, x, dout, h=1e-7):
    """
        computes numeric gradient of f at x using h (finite difference).
        
        Inputs:
            f: forward pass function
            x: torch tensor to compute gradient
            dout: upstream gradient
            h: finite differences
        Outputs:
            grad: the result of x gradient by finite approximation
    """
    x_flat = x.contiguous().flatten()
    downstream_grad = torch.zeros_like(x)
    downstream_grad_flat = downstream_grad.flatten()
    
    if dout is None:
        z = f(x)
        dout = torch.ones_like(z)

    upstream_grad = dout.flatten()
    
    for i in range(x_flat.shape[0]):
        old_val = x_flat[i].item()
        
        x_flat[i] = old_val + h
        fph = f(x)
        x_flat[i] = old_val - h
        fmh = f(x)

        x_flat[i] = old_val
        local_grad = (fph.flatten() - fmh.flatten()) / (2 * h)

        downstream_grad_flat[i] = torch.dot(upstream_grad, local_grad).item()
     
    return downstream_grad

def rel_error(x,y,eps=1e-10):
    """
        compute the relative error between x,y tensor
        
                                max_i |x_i - y_i]|
        rel_error(x, y) = -------------------------------
                        max_i |x_i| + max_i |y_i| + eps

        Inputs:
            x: Tensor
            y: Tensor
            eps: Small constant value for numeric stability
        Outputs:
            error: Scalar that compute the relative error between x and y
    """
    
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    error = top / bot
    return error
