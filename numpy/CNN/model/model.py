from utils.core import generate_batches, softmax_accuracy

class Model:
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def train(self, train_x, train_y, test_x, test_y, batch_size, epochs):

        for epoch in range(epochs):
            print(f"Starting Epoch: {epoch}")
            for idx, (batch_x, batch_y) in enumerate(generate_batches(train_x, train_y, batch_size)):
                if idx % 10 == 0:
                    print(f"{idx * batch_size} images has been trained")
                activation = self.forward_pass(batch_x, training=True)
                da_curr = activation - batch_y
                self.backward_pass(da_curr)
                self.update(epoch)

            prediction = self.forward_pass(test_x, training=False)
            accuracy = softmax_accuracy(prediction, test_y)
            print("accuracy: ", accuracy)
            
    def forward_pass(self, x, training):
        activation = x
        for layer in self.layers:
            activation = layer.forward_pass(activation, training)
        return activation
        
    def backward_pass(self, x):
        da_curr = x
        for layer in reversed(self.layers):
            da_curr = layer.backward_pass(da_curr)
    
    def update(self, epoch):
        self.optimizer.update(self.layers, epoch)
