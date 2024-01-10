EX 1.1
1. 0 if w * x + b <= 0 
2. 1 if w * x + b >  0

cw * cx + b <= 0  
    = c(w * x + b) <= 0\
    = w * x + b <= 0

we end up where we started, so the behavior of the network is unchanged. 

EX 1.2

1. 0 if w * x + b < 0 
2. 1 if w * x + b > 0
3. σ(z) ≡ 1 / (1 + e ** −z)
4. z = w * x + b

if we imagine that `w * x + b` is a positive number, as c approaches infinity, the output of the network approaches 1. 

1 / (1 + e ** -(cw * x + cb))\
    = 1 / (1 + e ** -c(w * x + b))\
    = 1 / (1 + e ** [c->∞]-c(w * x + b))\
    = 1 / (1 + 0)\
    = 1

Now, if we imagine that `w * x + b` is a negative number, as c approaches infinity, the output of the network approaches 0.     

1 / (1 + e ** -(cw * x + cb))\
    = 1 / (1 + e ** -c(w * x + b))\
    = 1 / (1 + e ** [c->∞]-c(w * x + b))\
    = 1 / (1 + ∞)\
    = 0

in both case, the behavior of the network is exactly same as the network of perceptrons.

special case: `w * x + b = 0`

1 / (1 + e ** -(cw * x + cb))\
    = 1 / (1 + e ** -c(w * x + b))\
    = 1 / (1 + e ** [c->∞]-c(w * x + b))\
    = 1 / (1 + e ** [c->∞]-c(0))\
    = 1 / (1 + 1)\
    = 1 / 2\
    = 0.5

0.5 != 0, so it fails. This can also be illustrated with `step function` vs sigmoid `logistic function`.

EX 2.1

An old output layer can activate several neurons in the new output layer.

The resulting output: 

0 -> [0,0,0,0]
1 -> [0,0,0,1]
2 -> [0,0,1,0]
3 -> [0,0,1,1]
4 -> [0,1,0,0]
5 -> [0,1,0,1]
6 -> [0,1,1,0]
7 -> [0,1,1,1]
8 -> [1,0,0,0]
9 -> [1,0,0,1]

Notice that the columns of above layout can be thought of as each neuron in the new output layer. In the reverse order, first column is the last neuron and the last column is the first neuron. The weight is either 0 or 1. The bias is 0 because we only care about the activation of the neuron.

For example, if we get `5` as an input from the previous layer, first and third neuron will be activated which in turn represents 5 (0101) in binary form.
