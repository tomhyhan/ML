import torch.nn as nn

class YOLOV1Lloss(nn.Module):
    def __init__(self,  S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        

    def forward(self, preds, targets, use_sigmoid=False):
        # define batch size
        
        # reshape preds
        
        # apply sigmoid to preds
        
        # create shifts for all grid cell locations normalize to 0 - 1
        
        # create grid for row and col
        
        # expand grid to (1, S, S, B)
        
        # extract pred_boxes from preds
        
        # get x1 y1 x2 y2 from xc_offset, yc_offset, w, h
        # x1 = xc - 0.5 * w
        
        # do the same for target
        
        pass