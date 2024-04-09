import numpy as np

def Sigmoid(Z):
    return 1 / 1 + np.exp(-Z)

def ReLU(Z):
    return np.maximum(0, Z)
