import numpy as np

def convert_one_hot(y, N_CLASSES):
    n_y = y.size
    one_hot = np.zeros((n_y,N_CLASSES))
    one_hot[np.arange(n_y), y] = 1
    return one_hot
