# Perceptrons

output: 
 1. 0 if w * x + b <= 0 
 2. 1 if w * x + b >  0

where w is the `weight` vector, b is the `bias` (inverse of threshold), and x in the `input` binary.

Interesting to note is that perceptrons can be thought if as a `NAND` logic gate with which any other logic gate can be constructed. 

# Sigmoid Neurons

The problem with perceptron is that the small change in the weight or bias can cause the output to flip from 0 to 1 or vice versa. Consequently, it becomes very hard to modify the weights and biases to get the desired behavior.

The Sigmoid Neurons was introduced to overcome this problem. The small changes in the weights and biases will only cause a small change in the output.

The importance of the σ function stems from the smoothenss of the logistic function 

<!-- come back to this -->
Partial Calculus

f(w,d) = o

do = dw + db

# The architecture of neural networks

A network is composed of input, output, and hidden layers each consists of neurons.

There are two types of neural networks: feedforward and recurrent.

# Gradient Descent

y(x) = y
y(x) = (0,0,0,0,0,0,1,0,0,0)**T

x is an input, y is a 10-dimensional vector, and T in the transpose operator.

Our `quadratic cost function`, also known as `mean squared error`::
- C(w,b) ≡ 1 / 2n \* ∑x ∥ y(x) − a ∥

w = weights
b = biases
n = number of training inputs
x = input (training data)
a = vector of outputs from the network when x is input. Depends on x, b and w.

As cost function approaches 0, the network is doing a good job. However, if the cost function is a large number, the large number of inputs are farther away from y(x), our desired output.

So, our goal is to find the weights and biases that minimize the cost function using `gradient descent`.

In order to find the global minimum of multi-variable function, we can use `Partial Differential Equation`
ex. 
- ΔC ≈ (∂C / ∂v1) * Δv1 + (∂C∂ / v2) * Δv2

Now, let's break down this form:

Δv ≡ (Δv1,Δv2)T
where Δv is a `vector of changes`.

∇C ≡ (∂C/∂v1,∂C/∂v2)T
where ∇C is a `gradient vector`

Then we can rewrite the equation as:
ΔC ≈ ∇C * Δv

∇C relates `the change in v to changes in C`

We're going to find a way of choosing Δv1 and Δv2 so as to make ΔC negative

1. Δv = (−) * η * ∇C
2. ΔC ≈ −η∇C * ∇C = −η * ∥∇C∥2
3. v → v′ = v − η∇C

η is the learning rate.

Third euqation basically says to move from v to v' in the `direction of the negative gradient` times the `learning rate`. In fact, this is the `gradient descent algorithm`.

Now, ΔC will always decrease

