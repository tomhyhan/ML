from data import load_data
from layers.conv2d import ConvLayer2D
from layers.pooling import MaxPoolLayer
from layers.relu import ReLULayer
from layers.flatten import FlattenLayer
from layers.fc import FullyConnectedLayer
from activation.softmax import SoftmaxLayer
from optimizer.adam import Adam
from model.model import Model
import numpy as np

train_data, train_target, test_data, test_target, valid_data, valid_target = load_data()

LAYERS = [
    ConvLayer2D.initialize(4, (3,3,1), padding="same"),
    ReLULayer(),
    MaxPoolLayer(pool_size=(2,2), stride=2),
    FlattenLayer(),
    FullyConnectedLayer.initialize(14 * 14 * 4, 64),
    ReLULayer(),
    FullyConnectedLayer.initialize(64, 10),
    SoftmaxLayer()
]

optimizer = Adam(lr=0.003)

model = Model(layers=LAYERS, optimizer=optimizer)

model.train(train_data, train_target, valid_data, valid_target, batch_size=64, epochs=1)

