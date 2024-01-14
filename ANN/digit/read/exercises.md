# EX 1.1
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

# EX 2.1

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

# EX 3.1 
<!-- COME BACK -->

It can be proved that the choice of Δv which minimizes ∇C⋅Δv is Δv=−η∇C
, where η= ϵ / ∥∇C∥ is determined by the size constraint ∥Δv∥=ϵ

∇C⋅Δv <= ∥∇C∥ ∥Δv∥

∇C⋅Δv is Δv=−η∇C
∥Δv∥ = ϵ => small fixed size move

∥Δv∥ = square root of Δv ' Δv

# EX 3.2

If we are dealing with a cost function with a single variable, the negative derivative of a function is a direction towards finding the minimum of the function. 

In geometric interpretation, we will have a single line describing our function. Choose a random point and start to move towards the minimum of the function by a fixed step.

# EX 3.3

One advantage of on-line learning is that the network can learn about new data without knowing the total number of traning data.

However, on-line learning can be very slow becuase the network has to compute the cost function for every training data.

Compare with stochastic gradient descent, the on-line learning can be more accurate, but it can be very slow.

# EX 4.1

w jk, k -> j 

basically, think of j as row and k as column.

a′ = σ( wa + b) =   σ(∑k(wjk * ajk) + b)

σ(z) ≡ 1 / (1 + exp(−z)) = 1 / (1 + exp(−∑k(wjk * ajk) + b))