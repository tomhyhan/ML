import math
import torch
import torch.utils
import torchvision
import torch.nn as nn


def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):
    # box_transform_pred.detach().reshape(-1, 1, 4)
    # (Batch_Size*H_feat*W_feat*Number of Anchors per location, 1, 4)
    # anchors: (H_feat * W_feat * num_anchors_per_location, 4)

    # reshape box_trainform_pred to (size(0), -1, 4)
    box_transform_pred = box_transform_pred.reshape(
        box_transform_pred.size(0), -1, 4
    )

    # define center x, center y. [10]
    # dx, dy, dw, dh -> [B*10, 1]
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + w / 2
    center_y = anchors_or_proposals[:, 1] + h / 2

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]
    # clamp dw, dh with max=math.log(1000.0 / 16)
    # can scale up to exp(4.1)
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    # define px, py, ph, pw
    px = dx * w[:, None] + center_x[:, None]
    py = dy * h[:, None] + center_y[:, None]
    pw = torch.exp(dw) * w[:, None]
    ph = torch.exp(dh) * h[:, None]

    # define p box using above
    pbox_x1 = px - 0.5 * pw
    pbox_y1 = py - 0.5 * ph
    pbox_x2 = px + 0.5 * pw
    pbox_y2 = py + 0.5 * ph

    pbox = torch.stack([pbox_x1, pbox_y1, pbox_x2, pbox_y2], dim=2)

    return pbox


def clamp_boxes_to_image_boundary(boxes, image_shape):
    H, W = image_shape.shape[-2:]

    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    x1 = x1.clamp(min=0, max=W)
    y1 = y1.clamp(min=0, max=H)
    x2 = x2.clamp(min=0, max=W)
    y2 = y2.clamp(min=0, max=H)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    return boxes


def get_iou(boxes1, boxes2):
    # param sample: gt_boxes, anchors
    # get area of each box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # get x1, y1 x2 y2from box1 and box2
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    # compute intersection area and union
    intersection_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = area1[:, None] + area2 - intersection_area

    # compute iou and return
    iou = intersection_area / union
    return iou


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super().__init__()
        self.scales = scales
        self.low_iou_threshold = model_config["rpn_bg_threshold"]
        self.high_iou_threshold = model_config["rpn_fg_threshold"]
        self.rpn_nms_threshold = model_config["rpn_nms_threshold"]
        self.rpn_batch_size = model_config["rpn_batch_size"]
        self.rpn_pos_count = int(
            model_config["rpn_pos_faction"] * self.rpn_batch_size)
        self.rpn_topk = model_config["rpn_train_topk"] if self.training else model_config["rpn_test_topk"]
        self.rpn_prenmns_topk = model_config["rpn_train_prenms_topk"] if self.training else model_config["rpn_test_prenms_topk"]
        self.aspact_ratios = aspect_ratios
        self.num_anchors = len(self.scales) * len(self.aspact_ratios)

        self.rpn_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        self.cls_layers = nn.Conv2d(in_channels, self.num_anchors, 1, 1, 0)

        self.bbox_reg_layer = nn.Conv2d(
            in_channels, self.num_anchors * 4, 1, 1, 0)

        for layer in [self.num_anchors, self.rpn_conv, self.cls_layers]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def generate_anchors(self, image, feat):
        # define grid h w, image h,w
        grid_h, grid_w = feat.shape[-2:]
        im_h, im_w = image.shape[-2:]

        # define stride with image and grid h,w
        stride_h = torch.tensor(
            im_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(
            im_w // grid_w, dtype=torch.int64, device=feat.device)

        # turn scales, aspect ratios to torch tensor
        aspact_ratios = torch.tensor(
            self.aspact_ratios, dtype=feat.dtype, device=feat.device)
        scales = torch.tensor(
            self.scales, dtype=feat.dtype, device=feat.device)

        # define h ratios and w ratios
        # tip: A = h*w and r = h/w
        # w = h/r | Ar = h**2
        h_ratios = aspact_ratios.sqrt()
        w_ratios = 1 / h_ratios

        # multiple scales with ratios with braodcast
        hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)
        ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)

        # zero center anchors (base_anchors)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()

        # define x-axis and scale by the stride. same you y-axis
        x_axis = torch.arange(0, grid_w, dtype=torch.int32,
                              device=feat.device) * stride_w
        y_axis = torch.arange(0, grid_h, dtype=torch.int32,
                              device=feat.device) * stride_h

        # create grid from x and y axis, reshape them to one dim tensor
        shift_y, shift_x = torch.metchgrid(y_axis, x_axis, index="ij")
        shift_y = shift_y.reshape(-1)
        shift_x = shift_x.reshape(-1)

        # stack shifts so that there are 4k coord x1 y1 x2 y2
        shift = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

        # add anchors to shifts with broadcast
        anchors = shift[:, None, 4] + base_anchors[None, :, 4]

        # reshape anchors to (-1, 4)
        anchors = anchors.reshape(-1, 4)
        # return anchors
        return anchors

    def filter_proposals(self, proposals, cls_scores, image_shape):
        # flatten cls score and apply sigmoid
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)

        # take topk btw prenms_topk and length if cls scores
        _, topk_indices = torch.topk(cls_scores, min(
            len(cls_scores, self.rpn_prenmns_topk)))

        # apply indices to cls scores and proposals
        cls_scores = cls_scores[topk_indices]
        proposals = proposals[topk_indices]

        # clamp boxes to image boundary
        # -> clamp_boxes_to_image_boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)

        # filter small boxes based on size 16
        min_size = 16
        w = proposals[..., 2] - proposals[..., 0]
        h = proposals[..., 3] - proposals[..., 1]
        keep = (w >= min_size) & (h >= min_size)

        # apply filtered indices to proposals and cls_scores
        cls_scores = cls_scores(keep)
        proposals = proposals(keep)

        # apply nms to proposals
        topk_indices = torchvision.ops.nms(
            proposals, cls_scores, self.rpn_nms_threshold)

        # sort keep indices by class and
        _, sorted_indices = torch.sort(
            cls_scores[topk_indices], descending=True)
        topk_indices = topk_indices[sorted_indices]

        # post nms topk filtering
        cls_scores = cls_scores[topk_indices]
        proposals = proposals[topk_indices]

        # return proposals and cls_scores
        return proposals, cls_scores

    def assign_targets_to_anchors(self, anchors, gt_boxes):

        pass

    def forward(self, image, feat, target=None):
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layers(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        anchors = self.generate_anchors(image, feat)

        batch_size = cls_scores.size(0)
        num_anchors_per_location = cls_scores.size(1)
        H, W = feat.shape[-2:]

        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)

        box_transform_pred = box_transform_pred.reshape(
            batch_size, num_anchors_per_location, 4, H, W
        )
        box_transform_pred = box_transform_pred.permute(
            0, 3, 4, 1, 2
        )
        box_transform_pred = box_transform_pred.reshape(-1, 4)

        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred.detach().reshape(-1, 1, 4), anchors
        )
        proposals = proposals.reshape(proposals.size(0), 4)
        # prediction boxes by matching pred to anchors

        proposals, scores = self.filter_proposals(
            proposals, cls_scores.detach(), image.shape)

        rpn_output = {
            "proposals": proposals,
            "scores": scores
        }

        if not self.training or target is None:
            return rpn_output
        else:
            # so far we have prediction anchor boxes

            pass


class FasterRCNN(nn.Module):
    def __init__(self, model_config, num_classes):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1]
        self.rpn = RegionProposalNetwork(
            model_config["backbone_out_channels"],
            scales=model_config["scales"],
            aspect_ratios=model_config["aspect_ratios"],
            model_config=model_config
        )

    def forward(self):
        pass
