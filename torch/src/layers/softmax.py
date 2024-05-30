import torch

class Softmax:
    @staticmethod
    def forward(x):
        """
            Computes forward pass for the softmax layer 
            
            Inputs:
                x: (N,D) result of network computation into number of classification
            Outputs:
                scores: softmax probability scores for each class 
        """
        # exp_x = x.clone()
        # exp_x -= torch.max(x, dim=1, keepdim=True).values
        # prob = torch.exp(exp_x) / torch.sum(torch.exp(exp_x), dim=1, keepdim=True)
        # scores = prob.clone() 
        
        
        shifted_logits = x - x.max(dim=1, keepdim=True).values
        Z = shifted_logits.exp().sum(dim=1, keepdim=True)
        log_probs = shifted_logits - Z.log()
        probs = log_probs.exp()
        
        return probs

    @staticmethod
    def backward(scores, y):
        """
            computes backward pass for softmax layer
            
            Inputs:
                scores: softmax probability scores for each class
                y: true label for data set
            Outputs:
                loss: Scalar giving the loss
                dx: downstream gradients w.r.t. x
        """
        N = scores.shape[0]
        loss = - (1 / N) * scores.clone()[torch.arange(N), y].log().sum() 
    
        dx = scores.clone()
        dx[torch.arange(N), y] -= 1
        dx /= N
        return loss, dx
    
    
