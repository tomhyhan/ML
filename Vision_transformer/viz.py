import matplotlib.pyplot as plt
import numpy as np

def draw_loss(training_loss, window=5):
    training_loss = np.array(training_loss)
    
    kernel = np.ones(window) / window
    moving_avg = np.convolve(training_loss, kernel, mode="valid")
    
    plt.figure(figsize=(8, 6))
    plt.plot(training_loss, label="Loss history", marker='o', color="blue")
    plt.plot(moving_avg, color="red")
    
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.title("Loss over Iterations")

    # plt.legend()
    # plt.gcf().set_size_inches(15, 25)
    plt.show()
    
def draw_train_val_accuracy(training_acc, val_acc):
    plt.figure(figsize=(8, 6))
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.plot(training_acc, label="Training Accuracy")
    plt.plot(val_acc, color="orange", label="Val Accuracy")
    
    
    plt.show()

