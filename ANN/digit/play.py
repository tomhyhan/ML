import math
import numpy as np

# t = 2
# print(math.log(2) * 2 ** t )
# print(math.log(2) * math.exp(math.log(2) * t))

# cross entropy when y is not 0 or 1
# y = 0.49999
# print((y * math.log(y)) + ((1 - y)*math.log((1-y))))

# print(math.exp(1))


# softmax limit value
# print(math.exp(10 * 1.2) / math.exp(10 * 1.3))
# print(math.exp(100 * 1.2) / math.exp(100 * 1.3))

# print(math.exp(1.2) / math.exp(1.2))

# x = np.array([0.4,0.5,0.6])
# y = np.array([0.1,0.2,0.3])

# print(np.linalg.norm(x-y))

# standard deviation of normal distribution
# squashing Gaussian down
# print(np.sqrt((1 / 1000) * 500 + 1))
# print(np.sqrt(501) / np.sqrt(1000))

# print(2 * math.sqrt(2))
# print(2 * math.sqrt(2/2))
# print(2 * math.sqrt(3/2))

# init weight
x = np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0/10), size=(5, 10)))
y = np.random.normal(loc=0.0, scale=np.sqrt(1.0/10), size=(5, 10))
print(x, )
print()
print(type(x) == type(y))
print(np.random.randn(2,2))