import torch
import torch.nn as nn

def get_iou(pred, target):
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    
    x1 = torch.max(pred[..., 0], target[...,0])
    y1 = torch.max(pred[..., 1], target[...,1])
    x2 = torch.min(pred[..., 2], target[...,2])
    y2 = torch.min(pred[..., 3], target[...,3])
    
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp()
    
    iou = inter / (pred_area.clamp() + target_area.clamp() - inter) + 1E-6
    return iou

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
        batch_size = preds.size(0)
        
        # reshape preds
        preds = preds.reshape(batch_size, self.S, self.S, self.B*5 + self.C)
        
        # apply sigmoid to preds
        if use_sigmoid:
            preds[..., :5*self.B] = nn.functional.sigmoid(preds[..., :5*self.B])
        
        # create shifts for all grid cell locations normalize to 0 - 1
        shift_x = torch.arange(0, self.S, device=preds.device, dtype=torch.int32) / float(self.S)
        shift_y = torch.arange(0, self.S, device=preds.device, dtype=torch.int32) / self.S
        
        # create grid for row and col
        shift_y, shift_x = torch.meshgrid(shift_x, shift_y, indexing="ij")
         
        # expand grid to (1, S, S, B)
        shift_y = shift_y.reshape(1, self.S, self.S, 1).repeat(1,1,1,self.B)
        shift_x = shift_x.reshape(1, self.S, self.S, 1).repeat(1,1,1,self.B)
        
        # extract pred_boxes from preds
        pred_boxes = preds[..., 5*self.B].reshape(batch_size, self.S, self.S, self.B, -1)
        
        # get x1 y1 x2 y2 from xc_offset, yc_offset, w, h
        # x1 = xc - 0.5 * w
        x1 = pred_boxes[..., 0] / self.S + shift_x - 0.5 * torch.square(pred_boxes[..., 2])
        x1 = x1[..., None]
        y1 = pred_boxes[..., 1] / self.S + shift_y - 0.5 * torch.square(pred_boxes[..., 3])
        y1 = y1[..., None]
        x2 = pred_boxes[..., 0] / self.S + shift_x + 0.5 * torch.square(pred_boxes[..., 2])
        x2 = x2[..., None]
        y2 = pred_boxes[..., 1] / self.S + shift_y + 0.5 * torch.square(pred_boxes[..., 3])
        y2 = y2[..., None]
        pred_x1y1x2y2 = torch.cat([x1, y1, x2, y2], dim=-1) 
        
        # do the same for target
        target_boxes = targets[..., 5*self.B].reshape(batch_size, self.S, self.S, self.B, -1)
        x1 = target_boxes[..., 0] / self.S + shift_x - 0.5 * torch.square(target_boxes[..., 2])
        x1 = x1[..., None]
        y1 = target_boxes[..., 1] / self.S + shift_y - 0.5 * torch.square(target_boxes[..., 3])
        y1 = y1[..., None]
        x2 = target_boxes[..., 0] / self.S + shift_x + 0.5 * torch.square(target_boxes[..., 2])
        x2 = x2[..., None]
        y2 = target_boxes[..., 1] / self.S + shift_y + 0.5 * torch.square(target_boxes[..., 3])
        y2 = y2[..., None]
        target_x1y1x2y2 = torch.cat([x1, y1, x2, y2], dim=-1) 

        iou = get_iou(pred_x1y1x2y2, target_x1y1x2y2)
        
        # get max iou val and idx
        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True)
        
        # create object indicator
        max_iou_idx = max_iou_idx.repeat(1,1,1,self.B)
        bb_idxs = torch.arange(self.B).reshape(1,1,1,self.B).expand_as(max_iou_idx)
        is_max_iou_box = (max_iou_idx == bb_idxs).long()
        
        obj_indicator = targets[..., 4:5]
        
        # classification loss
        cls_pred = preds[..., self.B*2:]
        cls_target = targets[..., self.B*2:]
        cls_mse = (cls_pred - cls_target) ** 2
        cls_mse = (obj_indicator * cls_mse).sum()
        
        # objectness loss
        # (B, S, S, 2)
        is_max_box_obj_indicator = is_max_iou_box * obj_indicator
        # (B, S, S, 1)
        obj_mse = (pred_boxes[..., 4] - max_iou_val)
        # (B, S, S, 2)
        obj_mse = (is_max_box_obj_indicator * obj_mse).sum()
        
        # localization loss
        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        x_mse = (is_max_box_obj_indicator * x_mse).sum()

        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        y_mse = (is_max_box_obj_indicator * y_mse).sum()

        w_sqrt_mse = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2
        w_sqrt_mse = (is_max_box_obj_indicator * w_sqrt_mse).sum()

        h_sqrt_mse = (pred_boxes[..., 3] - target_boxes[..., 3]) ** 2
        h_sqrt_mse = (is_max_box_obj_indicator * h_sqrt_mse).sum()
        
        # no object mse
        no_obj_indicator = 1 - is_max_box_obj_indicator
        no_obj_mse = (preds[..., 4] - torch.zeros_like(preds[..., 4])) ** 2
        no_obj_mse = (no_obj_mse * no_obj_indicator).sum()
        
        loss = self.lambda_coord*(x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)
        loss += cls_mse + obj_mse
        loss += self.lambda_noobj * no_obj_mse
        loss /= batch_size
        return loss
        
        