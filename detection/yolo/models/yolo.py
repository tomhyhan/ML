import torch.nn as nn
import torchvision

class YOLOV1(nn.Module):
    def __init__(self, im_size, num_classes, model_config):
        super().__init__()
        self.im_size = im_size
        self.im_channels = model_config["im_channels"]
        self.backbone_channels = model_config["backbone_chanels"]
        self.yolo_conv_channels = model_config["yolo_conv_channels"]
        self.conv_spatial_size = model_config["conv_spatial_size"]
        
        self.leaky_relu_slope = model_config["leaky_relu_slope"]
        self.yolo_fc_hidden_dim = model_config["fc_dim"]
        self.yolo_fc_dropout_prob = model_config["fc_dropout"]
        self.use_conv = model_config["use_conv"]
        self.S = model_config["S"]
        self.B = model_config["B"]
        self.C = num_classes
        
        backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )
        
        
        