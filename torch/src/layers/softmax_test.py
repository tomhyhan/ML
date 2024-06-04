import torch
import math
from softmax import Softmax
from src.test_tools.grads import grad_check_sparse, compute_numeric_gradient, rel_error


def test_loss():
    device = "cpu"
    dtype = torch.float64

    x = torch.randn(10, 10, device=device, dtype=dtype) * 0.01
    y = torch.randint(0, 10, (x.shape[0],), device=device)

    scores = Softmax.forward(x)
    loss, _ = Softmax.backward(scores, y)

    print("Sanity check for the initial loss value")
    print("expected:", math.log(x.shape[1]))
    print("result:", loss.item())


def test(x, y):
    scores = Softmax.forward(x)
    loss, _ = Softmax.backward(scores, y)
    return loss


def test_gradients():
    device = "cpu"
    dtype = torch.float64

    x = torch.randn(3073, 10, device=device, dtype=dtype) * 0.0001
    y = torch.randint(0, 10, (x.shape[0],), device=device)

    scores = Softmax.forward(x)
    loss, dx = Softmax.backward(scores, y)

    def fx(X): return test(X, y)

    grad_check_sparse(fx, x, dout=None, analytic_grads=dx, num_checks=10)

    # taking too much time
    # grad = compute_numeric_gradient(fx, x, dout=None)
    # error = rel_error(grad, dx)
    # print(error)


if __name__ == "__main__":
    test_loss()
    test_gradients()
