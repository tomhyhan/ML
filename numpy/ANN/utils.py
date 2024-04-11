import numpy as np

eps=1e-8

def Sigmoid(Z):
    return 1 / 1 + np.exp(-Z)

def Sigmoid_prime(z):
    return Sigmoid(z) * (1 - Sigmoid(z))

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_prime(Z):
    return np.where(Z > 0, 1, 0)

def cost_function(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) 

def loss_function(a, y):
    return - (np.divide(y,np.maximum(a,eps)) - np.divide(1-y, 1-np.maximum(a,eps)))