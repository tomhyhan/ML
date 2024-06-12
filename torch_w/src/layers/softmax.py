import torch

class Softmax:
    def __init__(self):
        pass
    
    def forward(self, X):
        """
            Computes forward pass for softmax layer
        """
        x = X.clone()
        sx = x - torch.max(x, dim=1, keepdim=True).values
        Z = sx.exp().sum(dim=1, keepdim=True)
        log_prob = sx - Z.log()
        scroes_prob = log_prob.exp()
        
        return scroes_prob
    
    def backward(self, scores, y):
        """
            Compute backpropagation for softmaxlayer
        """
        N = scores.shape[0]
        loss = - (1 / N) *  scores.clone()[torch.arange(N), y].log().sum() 
        
        dx = scores.clone()
        dx[torch.arange(N), y] -= 1
        dx /= N
        
        return loss,  dx
