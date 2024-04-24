import numpy as np

def convert_one_hot(y, N_CLASSES):
    n_y = y.size
    one_hot = np.zeros((n_y,N_CLASSES))
    one_hot[np.arange(n_y), y] = 1
    return one_hot

def generate_batches(x : np.array , y: np.array, batch_size: int):
    # print(x.shape)
    for i in range(0, x.shape[0], batch_size):
        # print(i, i+batch_size)
        # print(np.min(np.arange(0, 10)))
        # print(np.arange(i, np.min(i+batch_size, x.shape[0])))
        yield(
            x.take(indices=range(i, 
            min(i+batch_size, x.shape[0])), axis=0),
            y.take(indices=range(i, 
            min(i+batch_size, y.shape[0])), axis=0)
        )

def softmax_accuracy(y_hat, y):
    return np.sum(np.argmax(y_hat, axis=1) == y)
