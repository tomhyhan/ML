import torch
import  torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        im_channel,
        channels=[64,128,256],
        kernels=[4,4,4,4],
        strides=[2,2,2,1],
        paddings=[1,1,1,1]
    ):
        super().__init__()
        channels = [im_channel] + channels + [0]
        activation = nn.LeakyReLU(0.2)
        
        self.layers = nn.ModuleList()
        for i in range(len(channels)-1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=[i], stride=strides[i], padding=paddings[i], bias=False if i!=0 else True),
                    nn.BatchNorm2d(channels[i+1]) if i !=0 and i != len(channels) - 2 else nn.Identity(),
                    activation if i != len(channels)-2 else nn.Identity()
                )
            )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out 
        