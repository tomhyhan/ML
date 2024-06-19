import sys
sys.path.append("../")

import torch
from src.layers.basicblock import BasicBlock
from src.layers.bottleneck import Bottleneck
from src.layers.averagepool import AveragePool
from src.layers.fc import FullyConnectedLayer
from src.layers.softmax import Softmax

class ResNet:
    """
        Residual Neural Network
    """
    def __init__(self, block: BasicBlock | Bottleneck, n_blocks, device="cpu", base_width=64, groups=1, dtype=torch.float32) -> None:
        """
            Init required layers to train resnet
            
            Inputs:
                block: Block instance to use. BasicBlock or Bottleneck
                n_blocks: number of blocks for each blocks sharing same channels
                device: device 
                dtype:dtype
        """
        n_classes = 10
        self.base_width = base_width
        self.groups = groups
        self.device = device
        self.dtype = dtype
        
        self.C_in = 3
        self.block1 = self._define_blocks(64, block, n_blocks[0], stride=2)
        self.block2 = self._define_blocks(128, block, n_blocks[1],  stride=2)
        self.block3 = self._define_blocks(256, block, n_blocks[2],  stride=2)
        self.block4 = self._define_blocks(512, block,n_blocks[3],  stride=2)
        
        # test
        # self.C_in = 3
        # self.block1 = self._define_blocks(8, block, n_blocks[0], stride=2)
        # # N, 8, 16, 16
        # self.block2 = self._define_blocks(16, block, n_blocks[1],  stride=2)
        # # N, 16, 8, 8
        # self.block3 = self._define_blocks(32, block, n_blocks[2],  stride=2)
        # # N, 32, 4, 4
        # self.block4 = self._define_blocks(64, block,n_blocks[3],  stride=2)
        # # N, 64, 2, 2
        
        self.globalpool = AveragePool()
        
        # N, 64, 1, 1
        self.fc = FullyConnectedLayer(512 * block.expansion, n_classes, relu=False, device=device, dtype=dtype)
        self.softmax = Softmax()
        
        self.param_layers = [*self.block1, *self.block2, *self.block3, *self.block4, self.fc]
        
        
    def _define_blocks(self, C_out, block: BasicBlock, n_block, stride):
        blocks = []
        blocks.append(block(C_in=self.C_in, C_out=C_out, stride=stride, base_width=self.base_width, groups=self.groups, device=self.device, dtype=self.dtype))
        self.C_in = C_out * block.expansion
        for _ in range(1, n_block):
            blocks.append(block(C_in=self.C_in, C_out=C_out, stride=1, base_width=self.base_width, groups=self.groups, device=self.device, dtype=self.dtype))
        
        return blocks
        
        
    def loss(self, X, y=None):
        """
            Complete the forward pass and back propagation cycle. Compute the loss and gradients
            
            Inputs:
                X: input data
                y: True label for input data
            Outputs:
                loss: Scalar loss value
        """
        out = X
        # print(self.block1[0].conv1.w.shape)
        for bidx, blocks in enumerate([self.block1, self.block2, self.block3, self.block4]):        
            for block in blocks:
                out = block.forward(out)
        out = self.globalpool.forward(out)
        out = self.fc.forward(out)
        scores = self.softmax.forward(out)
        
        if y is None:
            return scores
        
        loss, dout = self.softmax.backward(scores, y)
        dout = self.fc.backward(dout)
        dout = self.globalpool.backward(dout)
        
        for blocks in reversed([self.block1, self.block2, self.block3, self.block4]):        
            for block in reversed(blocks):
                dout = block.backward(dout)
                
        return loss 
                
        
        
                                         
