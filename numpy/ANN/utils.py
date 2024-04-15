import numpy as np

eps=1e-8

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Sigmoid_prime(z):
    return Sigmoid(z) * (1 - Sigmoid(z))

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_prime(Z):
    return np.where(Z > 0, 1, 0)

def cost_function(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) 

def softmax(z):
    # print(z.shape)
    if z.ndim == 2:
        e = np.exp(z - np.max(z, axis=0, keepdims=True))
        sm = e / np.sum(e, axis=0, keepdims=True)
        return sm
    else:
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        sm = e / np.sum(e, axis=1, keepdims=True)
        return sm

def loss_function(a, y):
    return - (np.divide(y,np.maximum(a,eps)) - np.divide(1-y, 1-np.maximum(a,eps)))

def simple_loss_function(a, y):
    return a - y

def convert_one_hot(n):
    hot = np.zeros((10,1))
    hot[n] = 1.0
    return hot



# x = [1,2,3,4,5]
# x = softmax([x])
# print(x)
# print(softmax_cross_entropy(x, [[0,0,1,0,0]]))

# print(np.array([[0,0,1]]) * np.array([[5,5,5]]))
# print(np.log(0.6))