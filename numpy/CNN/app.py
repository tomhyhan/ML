from data import load_data
from layers.conv2d import ConvLayer2D
from layers.pooling import MaxPoolLayer
from layers.relu import ReLULayer
from layers.flatten import FlattenLayer
from layers.fc import FullyConnectedLayer
from activation.softmax import SoftmaxLayer
from optimizer.adam import Adam
from model.model import Model
from model.sequential import SequentialModel
import numpy as np

train_data, train_target, test_data, test_target, valid_data, valid_target = load_data()

LAYERS = [
    FlattenLayer(),
    FullyConnectedLayer.initialize(28 * 28, 64),
    ReLULayer(),
    FullyConnectedLayer.initialize(64, 10),
    SoftmaxLayer()
]

# print(train_data.shape, train_target.shape, valid_data.shape, valid_target.shape)
optimizer = Adam(lr=0.003)

# model = Model(layers=LAYERS, optimizer=optimizer)
model = Model(layers=LAYERS, optimizer=optimizer)

model.train(train_data, train_target, valid_data, valid_target, 1, 10)

