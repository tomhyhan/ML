from ..utils import generate_batches

class Model:
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def train(self, train_x, train_y, test_x, test_y, lr, batch_size, epochs):

        for epoch in range(epochs):
            mini_batches = generate_batches(train_x, train_y, batch_size)
            
            for mini_batch in mini_batches:
                pass
            pass
