import numpy as np
import matplotlib.pyplot as plt

def viz_loss_history(loss_history):
    plt.title("Training loss")
    plt.xlabel("Iteration")
    plt.plot(loss_history, marker='o', label="kaiming")
    moving_average = np.convolve(loss_history, np.ones(10)/ 10, mode="valid")
    plt.plot(moving_average, color="red",)
    plt.gcf().set_size_inches(15, 25)
    plt.show()
    
    
def viz_training_and_val(training, val):
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(training, label="kaiming")
    plt.plot(val, color="red",)
    plt.gcf().set_size_inches(15, 25)
    plt.show()