from utils.core import generate_batches, softmax_accuracy

class Model:
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def train(self, train_x, train_y, test_x, test_y, batch_size, epochs):

        for epoch in range(epochs):
            mini_batches = generate_batches(train_x, train_y, batch_size)

            print(f"Starting Epoch: {epoch}")
            for idx, (x, y) in enumerate(mini_batches):
                y_hat = self.forward_pass(x, training=True)
                activation = y_hat - y
                self.backward_pass(activation)
                self.update(epoch)
        
        y_hat = self.forward_pass(test_x)
        accuracy = softmax_accuracy(y_hat, test_y)
        print(f"accuracy: {accuracy} / {len(test_y)}")
            
    def forward_pass(self, x, training):
        for layer in self.layers:
            x = layer.forward_pass(x, training)
        return x 

    def backward_pass(self, da_curr):
        for layer in reversed(self.layers):
            da_curr = layer.backward_pass(da_curr)

    def update(self, epoch):
        self.optimizer.update(self.layers, epoch)
