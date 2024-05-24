import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)

def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W, dtype=torch.float32)


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    n_train = X.shape[0]
    scores = torch.matmul(X,W)
    max_scores, _ = torch.max(scores, dim=1, keepdim=True)
    scores = scores - max_scores
    scores = torch.exp(scores) / torch.sum(torch.exp(scores), dim=1, keepdim=True)
    # print(scores.device)
    # print(scores.dtype)
    # print(y.dtype)
    loss = -torch.sum(torch.log(scores[torch.arange(n_train), y]))

    # 128 10
    # 128 3063
    # print(X.shape)
    scores[torch.arange(n_train), y] -= 1
    delta = scores
    dW = torch.matmul(X.T, delta)
    # print(scores[0])

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss /= n_train
    loss += reg * torch.sum(W * W)

    dW /= n_train
    dW += 2*reg*W

    return loss, dW


def compute_grad(W, X, y, reg):
    
    # Define a loss function (using functional version)
    scores = torch.matmul(X,W)
    # y_pred = torch.nn.functional.softmax(scores, dim=1)
    loss = torch.nn.functional.cross_entropy(scores, y)
    # Perform backward pass to calculate gradients
    loss.backward()
    
    return loss, x.grad 


def compute_numeric_gradient(f, x, dLdf=None, h=1e-7):
    """
    Compute the numeric gradient of f at x using a finite differences
    approximation. We use the centered difference:

    df    f(x + h) - f(x - h)
    -- ~= -------------------
    dx           2 * h

    Function can also expand this easily to intermediate layers using the
    chain rule:

    dL   df   dL
    -- = -- * --
    dx   dx   df

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor giving the point at which to compute the gradient
    - dLdf: optional upstream gradient for intermediate layers
    - h: epsilon used in the finite difference calculation
    Returns:
    - grad: A tensor of the same shape as x giving the gradient of f at x
    """
    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    # Initialize upstream gradient to be ones if not provide
    if dLdf is None:
        y = f(x)
        dLdf = torch.ones_like(y)
    dLdf = dLdf.flatten()

    # iterate over all indexes in x
    for i in range(flat_x.shape[0]):
        oldval = flat_x[i].item()  # Store the original value
        flat_x[i] = oldval + h  # Increment by h
        fxph = f(x).flatten()  # Evaluate f(x + h)
        flat_x[i] = oldval - h  # Decrement by h
        fxmh = f(x).flatten()  # Evaluate f(x - h)
        flat_x[i] = oldval  # Restore original value

        # compute the partial derivative with centered formula
        dfdxi = (fxph - fxmh) / (2 * h)

        # use chain rule to compute dLdx
        flat_grad[i] = dLdf.dot(dfdxi).item()

    # Note that since flat_grad was only a reference to grad,
    # we can just return the object in the shape of x by returning grad
    return grad

# print("analytics", compute_grad(x))
# print("numeric", compute_numeric_gradient(compute_grad, torch.tensor([1., 2., 3.])))

W = torch.randn(10, 5, dtype=torch.float32, requires_grad=True)
X_batch = torch.rand(1, 10, dtype=torch.float32)
y_batch = torch.tensor([1], dtype=torch.int64)

loss, grad = softmax_loss_vectorized(W, X_batch, y_batch, 0)

loss1, grad1 = compute_grad(W, X_batch, y_batch, 0)

print(loss)
print(loss1)
