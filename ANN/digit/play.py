import math

t = 2
print(math.log(2) * 2 ** t )
print(math.log(2) * math.exp(math.log(2) * t))

# cross entropy when y is not 0 or 1
y = 0.49999
print((y * math.log(y)) + ((1 - y)*math.log((1-y))))

# print(math.exp(1))