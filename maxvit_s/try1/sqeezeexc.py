from torch import nn

class SqeezeExciation(nn.Module):
    def __init__(
        self,
        in_channel,
        sqeeze_channel,
        activation
        
    ) :
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=in_channel, out_channels=sqeeze_channel, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=sqeeze_channel,
                              out_channels=in_channel, kernel_size=1)
        self.activation = activation
        self.scale_activation = nn.Sigmoid()
        
    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale