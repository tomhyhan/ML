import numpy as np

eps=1e-8

def Sigmoid(z):
    # print(z)
    # if (1 + np.exp(-z)).any():
    #     print(z)
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
  e = np.exp(z - np.max(z))
  return e / np.sum(e, axis=0)

# Example usage
z = np.array([1, 2, 3, 4, 5])
scores = softmax(z)
print(scores)


def loss_function(a, y):
    return - (np.divide(y,np.maximum(a,eps)) - np.divide(1-y, 1-np.maximum(a,eps)))

def simple_loss_function(a, y):
    return a - y

def convert_one_hot(n):
    hot = np.zeros((10,1))
    hot[n] = 1.0
    return hot