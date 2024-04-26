import numpy as np

x, y = np.indices((5,8))
k = np.zeros((5,8))
# print(x)
# print()
# print(y)
k[:,:] = 2
print(k)