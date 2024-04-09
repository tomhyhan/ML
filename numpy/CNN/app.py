from tensorflow.keras.datasets import fashion_mnist
import numpy as np

N_TRAIN_SAMPLES = 50000
N_TEST_SAMPLE = 2500
N_VALID_SAMPLES = 250
N_CLASSES = 10
IMAGE_SIZE = 28

((train_x, train_y), (test_x, test_y))  = fashion_mnist.load_data()

train_data = train_x[:N_TRAIN_SAMPLES, :, :]
train_target = train_y[:N_TRAIN_SAMPLES]

test_data = train_x[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE, :, :]
test_target = train_y[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE]

valid_data = test_x[:N_VALID_SAMPLES, :, :]
valid_target = test_y[:N_VALID_SAMPLES]


print(train_data.shape, test_data.shape, valid_data.shape)
print((train_data / 255)[0])
print((train_data)[0])