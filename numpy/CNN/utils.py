import numpy as np

def convert_one_hot(y, N_CLASSES):
    n_y = y.size
    one_hot = np.zeros((n_y,N_CLASSES))
    one_hot[np.arange(n_y), y] = 1
    return one_hot

def generate_batches(x : np.array , y: np.array, batch_size: int):
    for i in range(0, x.shape[0], batch_size):
        x.take
        yield(
            x.take(np.arange(i, 
            np.min(i+batch_size, x.shape[0])), axis=0),
            y.take(np.arange(i, 
            np.min(i+batch_size, y.shape[0])), axis=0)
        )
