import torch
import numpy

"""
    imagine we have 2 independent weight dices
"""
# x = torch.tensor([10, 20 ,30 ,40, 50, 60])
# x = torch.tensor([6, 5 ,4 ,3, 2, 1])
# y = torch.tensor([1, 2, 6, 5, 3, 1])

# x -= x.max()
# Zx = x.exp().sum()
# probx = (x-Zx.log()).exp()

# y -= y.max()
# Zy = y.exp().sum()
# proby = (y-Zy.log()).exp()

x = [.12, .23, .31, .18, .12, .04]
y = [0.41, 0.25, 0.15, .1, .06, .03]

DP = [[0 for _ in range(len(x))] for _ in range(len(y))]
acc = [0] * (len(x) * 2 -1)
for i in range(len(x)):
    for j in range(len(y)):
        DP[i][j] = x[i] * y[j]
        acc[i+j] += x[i] * y[j]
             
print([f"{a:.3f}" for a in acc])

print([f"{a:.3f}" for a in numpy.convolve(x,y)])

"""
    As shown above, there are two ways compute probability distribution of x + y. 
    
        1. using multiplication table
        2. using convolution
        
    Now, the general formula for computing discrete probability for two independent variables:
    
    P(x + y) = (Px * Py)[s] = SUM (x:1->6)[Px(x) * Py(s-x)]
    
    ex:
        s = 4
        p(1) * P(3)
        p(2) * P(2)
        p(3) * P(1)
        p(4) * P(0) => 0
        p(5) * P(-1) => 0
        p(6) * P(-2) => 0
"""
pxy = 0
s = 4
for i in range(1, 7):
    m = s - i
    if m ==0:break
    pxy += DP[i-1][m-1] 
print(f"{pxy:.3f}")