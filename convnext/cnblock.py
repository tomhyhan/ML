class CNBlock:
    pass

class CNBlockConfig:
    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers,
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        
        