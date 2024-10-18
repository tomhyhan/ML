from torch import nn 

class SqeezeExciation(nn.Module):
    def __init__(
        self,
        input_channels,
        squeeze_channels,
        activation = nn.ReLU,
        scale_activation = nn.Sigmoid
    ):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()
        
    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale