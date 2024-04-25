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
                if idx % 1000 == 0:
                    print("images trained: ", idx * batch_size)
                y_hat = self.forward_pass(x, training=True)
                activation = y_hat - y
                self.backward_pass(activation)
                self.update(epoch)
                # break
            y_hat = self.forward_pass(test_x, training=False)
            accuracy = softmax_accuracy(y_hat, test_y)
            print(f"accuracy: {accuracy} / {len(test_y)}")
            
    def forward_pass(self, x, training):
        activation = x
        for idx, layer in enumerate(self.layers):
            activation = layer.forward_pass(activation, training)
        return activation

    def backward_pass(self, x):
        activation = x
        for layer in reversed(self.layers):
            activation = layer.backward_pass(da_curr=activation)

    def update(self, epoch):
        self.optimizer.update(self.layers, epoch)
