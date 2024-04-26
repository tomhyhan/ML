import numpy as np

def convert_one_hot(y, N_CLASSES):
    one_hot_matrix = np.zeros((y.size, y.max() + 1))
    one_hot_matrix[np.arange(y.size), y] = 1
    return one_hot_matrix

def generate_batches(x : np.array , y: np.array, batch_size: int):
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(
                i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(
                i, min(i + batch_size, y.shape[0])), axis=0)
        )

def softmax_accuracy(y_hat, y):
    class_idx = np.argmax(y_hat, axis=1)
    one_hot_matrix = np.zeros_like(y_hat)
    one_hot_matrix[np.arange(y_hat.shape[0]), class_idx] = 1
    y_hat = one_hot_matrix
    return (y_hat == y).all(axis=1).mean()
