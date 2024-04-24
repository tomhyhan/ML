import numpy as np
da_curr = np.array([[1,2,3], [4,5,6]])
a_prev = np.array([[1,2,3,4,5]])
weight = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
# print(da_curr.shape, weight.shape)
# print(np.dot(da_curr, weight))
print(np.sum(da_curr, axis=0, keepdims=True))
# print(np.dot(a_prev.T, da_curr).shape)