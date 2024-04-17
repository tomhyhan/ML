from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from utils import convert_one_hot


N_TRAIN_SAMPLES = 5000
N_TEST_SAMPLE = 1000
N_VALID_SAMPLES = 250
N_CLASSES = 10
IMAGE_SIZE = 28

def convert_x(t_x):
    return np.array([x.reshape(IMAGE_SIZE,IMAGE_SIZE,1) / 255 for x in t_x])

def load_data():
    ((train_x, train_y), (test_x, test_y))  = fashion_mnist.load_data()

    train_data = convert_x(train_x[:N_TRAIN_SAMPLES])
    train_target = convert_one_hot(train_y[:N_TRAIN_SAMPLES], N_CLASSES)

    test_data = convert_x(train_x[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE]) 
    
    test_target = convert_one_hot(train_y[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE], N_CLASSES)

    valid_data = convert_x(test_x[:N_VALID_SAMPLES])
    valid_target = convert_one_hot(test_y[:N_VALID_SAMPLES], N_CLASSES)

    return train_data, train_target, test_data, test_target, valid_data, valid_target
    # print(train_data.shape)
    # print(train_target.shape)

    # print(test_data.shape)
    # print(test_target.shape)

    # print(valid_data.shape)
    # print(valid_target.shape)