import torch
import math
from softmax import Softmax
from grads import grad_check_sparse,compute_numeric_gradient, rel_error

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx



def test_loss():
    device = "cpu"
    dtype = torch.float64
    
    x = torch.randn(10,10, device=device, dtype=dtype) * 0.01 
    y = torch.randint(0, 10, (x.shape[0],), device=device)
    
    scores = Softmax.forward(x)
    loss, _ = Softmax.backward(scores, y)
    
    print("Sanity check for the initial loss value")
    print("expected:", math.log(x.shape[1]))
    print("result:", loss.item())
    
def test(x,y):
    scores = Softmax.forward(x)
    loss,_ = Softmax.backward(scores, y)
    return loss

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    x = torch.randn(3073,10, device=device, dtype=dtype) * 0.0001
    y = torch.randint(0, 10, (x.shape[0],), device=device)
    
    scores = Softmax.forward(x)
    loss, dx = Softmax.backward(scores, y)

    fx = lambda X: test(X,y)
    
    grad_check_sparse(fx,x,dout=None, analytic_grads=dx, num_checks=10)
    
    # taking too much time
    # grad = compute_numeric_gradient(fx, x, dout=None)
    # error = rel_error(grad, dx)
    # print(error)
    
if __name__ == "__main__":
    test_loss()
    test_gradients()
    
