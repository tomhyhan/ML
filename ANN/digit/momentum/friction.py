import numpy as np

# Define the function to minimize (example: f(x) = x^2)
def f(x):
    return x**2

# Define the gradient of the function (example: f'(x) = 2*x)
def grad_f(x):
    return 2*x

# Gradient Descent function
def gradient_descent(lr, epochs, initial_x):
    x = initial_x
    for _ in range(epochs):
        gradient = grad_f(x)
        print("gradient: ", lr * gradient)
        x = x - lr * gradient
    return x

# Set learning rate and number of epochs
learning_rate = 0.1
num_epochs = 100

# Set initial value for x
initial_x = 5

# Run gradient descent
result = gradient_descent(learning_rate, num_epochs, initial_x)

print("Result:", result)

import numpy as np

# Define the function to minimize (example: f(x) = x^2)
def f(x):
    return x**2

# Define the gradient of the function (example: f'(x) = 2*x)
def grad_f(x):
    return 2*x

# Momentum Gradient Descent function (Polyak's Heavy Ball Method)
def polyak_momentum_gradient_descent(lr, epochs, initial_x, b):
    x = initial_x
    v = 0  # Initialize velocity to zero
    for _ in range(epochs):
        gradient = grad_f(x)
        v = b * v + (1 - b) * lr * gradient
        print("velocity", v)
        x = x - v
    return x

# Set learning rate, number of epochs, and momentum parameter
learning_rate = 0.1
b = 0.9

# Set initial value for x
initial_x = 5

# Run Polyak's Heavy Ball Method
result = polyak_momentum_gradient_descent(learning_rate, num_epochs, initial_x, b)

print("Result:", result)
