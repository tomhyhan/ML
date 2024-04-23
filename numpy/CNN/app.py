from data import load_data
from layers.conv2d import ConvLayer2D
from layers.pooling import MaxPoolLayer
from layers.relu import ReLULayer

train_data, train_target, test_data, test_target, valid_data, valid_target = load_data()

# conv layer, fc layer, relu, softmax
LAYERS = [
    ConvLayer2D.initialize(32, (3,3,1), padding="same", stride=1),
    ReLULayer(),
    MaxPoolLayer((2,2), stride=2)
]