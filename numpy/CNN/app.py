from data import load_data
from layers.conv2d import ConvLayer2D
from layers.pooling import MaxPoolLayer
from layers.relu import ReLULayer
from layers.flatten import FlattenLayer
from layers.fc import FullyConnectedLayer
from activation.softmax import SoftmaxLayer
from optimizer.adam import Adam
from model.model import Model

train_data, train_target, test_data, test_target, valid_data, valid_target = load_data()

# conv layer, fc layer, relu, softmax
LAYERS = [
    ConvLayer2D.initialize(32, (3,3,1), padding="same", stride=1),
    ReLULayer(),
    MaxPoolLayer((2,2), stride=2),
    FlattenLayer(),
    FullyConnectedLayer.intialize(14 * 14 * 32, 10),
    SoftmaxLayer()
]

optimizer = Adam(lr=0.001)

model = Model(layers=LAYERS, optimizer=optimizer)

model.train(train_data, train_target, valid_data, valid_target, 10, 1)

