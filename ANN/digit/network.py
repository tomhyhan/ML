import numpy as np

class Network:
    # net.weights[1] => w
    # W jk k -> j 
    # k is src and j is dest
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.rand(y,1) for y in sizes[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        
        
print(np.random.rand(3,10))