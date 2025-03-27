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


def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    xc = anchors_or_proposals[:, 0] + 0.5 * w
    yc = anchors_or_proposals[:, 0] + 0.5 * h

    gw = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gh = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gxc = ground_truth_boxes[:, 0] + 0.5 * w
    gyc = ground_truth_boxes[:, 1] + 0.5 * h

    tx = (gxc - xc) / w
    ty = (gyc - yc) / h
    tw = torch.log(gw / w)
    th = torch.log(gh / h)
    regression_targets = torch.stack([tx, ty, tw, th], dim=1)
    return regression_targets


def sample_positive_negative(labels, positive_count, total_count):
    # get positive and negative label indices
    positive = torch.where(labels > 0)[0]
    negative = torch.where(labels == 0)[0]

    # get number of positive and negative
    num_pos = torch.min(positive_count, positive.numel())
    num_neg = total_count - num_pos
    num_neg = torch.min(num_neg, negative.numel())

    # perm and resize by number of positive and negatives
    pos_idx = torch.randperm(
        positive.numel(), device=positive.device)[:num_pos]
    neg_idx = torch.randperm(
        negative.numel(), device=negative.device)[:num_neg]

    # create mask with label and apply the pos and neg indices
    perm_positive_idxs = torch.zeros_like(labels, dtype=torch.bool)
    perm_negative_idxs = torch.zeros_like(labels, dtype=torch.bool)

    perm_positive_idxs[pos_idx] = True
    perm_negative_idxs[neg_idx] = True

    return perm_positive_idxs, perm_negative_idxs


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
            len(cls_scores, self.rpn_prenmns_topk))
        )

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
        # get_iou from gt_boxes and anchor boxes
        iou_mattrix = get_iou(gt_boxes, anchors)

        # for each anchor box, assign gt box with max iou
        best_match_gt, best_match_gt_idx = iou_mattrix.max(dim=0)

        # create copy of best_match_gt_idx to best_match_gt_idx_pre_thresholding
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.copy()

        # assign -1 to background and -2 to neutral
        bg = best_match_gt < self.low_iou_threshold
        best_match_gt_idx[bg] = -1

        neutral = self.low_iou_threshold <= best_match_gt < self.high_iou_threshold
        best_match_gt_idx[neutral] = -2

        # for each gt box assign anchor box in max iou
        best_anchor_iou_for_gt, _ = iou_mattrix.max(dim=1)

        # we don't want to overlook anchors boxes with same max iou assign to a single gt box
        # so we compute indices of anchor boxes that have max iou to a single gt box without
        # considering the threshold
        _, gt_pred_pair_with_highest_iou = torch.where(
            iou_mattrix == best_anchor_iou_for_gt[:, None])

        # restore max iou assign to gt_boxes best_math_gt_idx from previous
        best_match_gt_idx[gt_pred_pair_with_highest_iou] = best_match_gt_idx_pre_thresholding[gt_pred_pair_with_highest_iou]

        # define matched_gt_boxes from best_match_gt_idx while clamp to 0
        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]

        # create labels, fg as 1 and cast to float 32
        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)

        # set bg as 0
        bg = best_match_gt_idx == -1
        labels[bg] = 0.0

        # set ignored as -1
        ignored = best_match_gt_idx == -2
        labels[ignored] = -1.0

        # return labels and match gt boxes
        return labels, matched_gt_boxes

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
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors, target["bboxes"][0]
            )

            regression_targets = boxes_to_transformation_targets(
                matched_gt_boxes_for_anchors, anchors
            )

            sampled_neg_idx_mask, sample_pos_idx_mask = sample_positive_negative(
                labels_for_anchors,
                positive_count=self.rpn_pos_count,
                total_count=self.rpn_batch_size
            )

            sampled_idx = torch.where(
                sampled_neg_idx_mask | sample_pos_idx_mask)[0]

            localization_loss = nn.functional.smooth_l1_loss(
                box_transform_pred[sample_pos_idx_mask],
                regression_targets[sample_pos_idx_mask],
                beta=1/9,
                reduction="sum"
            ) / sampled_idx.numel()

            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                cls_scores[sampled_idx].flatten(),
                labels_for_anchors[sampled_idx].flatten()
            )

            rpn_output["rpn_classification_loss"] = cls_loss
            rpn_output["rpn_localization_loss"] = localization_loss
            return rpn_output


class ROIHead(nn.Module):
    def __init__(self, model_config, num_classes, in_channels):
        super().__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config["roi_batch_size"]
        self.roi_pos_count = int(
            model_config["roi_pos_fraction"] * self.roi_batch_size)
        self.iou_threshold = model_config["iou_threshold"]
        self.low_bg_iou = model_config["roi_low_bg_iou"]
        self.nms_threshold = model_config["roi_nms_threshold"]
        self.topK_detections = model_config["roi_topk_detections"]
        self.low_score_threshold = model_config["roi_score_threshold"]
        self.pool_size = model_config["roi_poo_size"]
        self.fc_inner_dim = model_config["fc_inner_dim"]

        self.fc6 = nn.Linear(in_channels * self.pool_size *
                             self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)

        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, 4*self.num_classes)

        nn.init.normal_(self.cls_layer.weight, std=0.01)
        nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.bbox_reg_layer.weight, std=0.01)
        nn.init.constant_(self.bbox_reg_layer.bias, 0)

    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        # get_iou
        iou_matrix = get_iou(gt_boxes, proposals)

        # find best matching gt box, bg and ignored
        _, best_match_gt_idx = iou_matrix.max(dim=0)
        bg = self.low_bg_iou <= best_match_gt_idx < self.iou_threshold
        ignored = best_match_gt_idx < self.low_bg_iou

        # assign bg to -1, ignored to -2
        best_match_gt_idx[bg] = -1
        best_match_gt_idx[ignored] = -2

        # get match_gt_boxes_for_proposals
        match_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]

        # get class labels
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)

        # assign 0 to bg, -1 to ignored
        labels[bg] = 0
        labels[ignored] = -1

        # return labels, matched gt boxes
        return labels, match_gt_boxes_for_proposals

    def forward(self, feat, proposals, image_shape, target):

        if self.training and target is not None:
            proposals = torch.cat([proposals, target['bboxes'][0]], dim=0)
            gt_boxes = target["bboxes"][0]
            gt_labels = target["labels"][0]

            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(
                proposals, gt_boxes, gt_labels
            )

            # sample indices
            neg_sample_idx, pos_sample_idx = sample_positive_negative(
                labels, self.roi_pos_count, self.roi_batch_size)
            sample_idx = torch.where(neg_sample_idx | pos_sample_idx)[0]

            # keep sampled proposals, labels, match gt boxes
            proposals = proposals[sample_idx]
            labels = labels[sample_idx]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sample_idx]

            # get box regression targets
            regression_targets = boxes_to_transformation_targets(
                matched_gt_boxes_for_proposals, proposals)
            
        # get scale from original image to feat
        feat_shape = feat.shape[-2:]
        scales = []
        for fs, ms in zip(feat_shape, image_shape):
            scale = float(fs) / float(ms)
            scale = 2 ** float(torch.tensor(scale).log2().round())
            scales.append(scale)

        # apply roi pooling for proposals
        proposal_roi_pool_feats = torchvision.ops.roi_pool(feat, [proposals], self.pool_size, scales[0])
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        box_fc_6 = nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = nn.functional.relu(self.fc7(box_fc_6))
        
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        
        # reshape box transform pred
        num_proposals, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_proposals, num_classes, 4)
        
        frcnn_output = {}
        if self.training or target is not None:
            # get classification loss
            classification_loss = nn.functional.cross_entropy(cls_scores, labels)
        
            # get foreground idxs and labels
            fg_idxs = labels > 0
            fg_box_transform_pred = box_transform_pred[fg_idxs]
            fg_labels = labels[fg_idxs]
                                    
            # compute localization loss
            localization_loss = nn.functional.smooth_l1_loss(
                fg_box_transform_pred[fg_idxs, fg_labels],
                regression_targets[fg_idxs],
                beta=1/9,
                reduction="sum"
            )
            
            localization_loss = localization_loss / labels.numel()
            frcnn_output["frcnn_classification_loss"]  = classification_loss
            frcnn_output["frcnn_localization_loss"] = localization_loss    
        
        if self.training:
            return frcnn_output
        else:
            device = cls_scores.device
            # apply transformation pred to proposals
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(box_transform_pred, proposals)
            pred_scores = nn.functional.softmax(cls_scores, dim=-1)
                        
            # clamp box to image boundary
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)
            
            # create labels for each prediction
            pred_labels = torch.arange(num_classes, device=device)
            pred_labels = pred_labels.view(-1, 1).expand_as(pred_scores)
            
            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]
            
            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)
                        
            pred_boxes, pred_scores, pred_labels = self.filter_predictions(pred_boxes, pred_scores, pred_labels)
            frcnn_output["boxes"] = pred_boxes
            frcnn_output["scores"] = pred_scores
            frcnn_output["labels"] = pred_labels
            return frcnn_output
        
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        # filter low scoring boxes
        keep = pred_scores < self.low_score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        
        # remove small size boxes
        min_size = 16
        ws = pred_boxes[:, 2] - pred_boxes[:, 0]
        hs = pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws > min_size) & (hs > min_size)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        
        # nms for each class
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for cls in range(torch.unique(pred_labels)):
            cls_keep = cls == pred_labels
            curr_keep_indices = torchvision.ops.nms(
                pred_boxes[cls_keep], pred_scores[cls_keep], self.nms_threshold
            )
            keep_mask[cls_keep[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indicies = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
        keep = post_nms_keep_indicies[:self.topK_detections]
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        return pred_boxes, pred_labels, pred_scores
    
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
        self.roi_head = ROIHead(
            model_config,
            num_classes,
            in_channels=model_config["backbone_out_channels"]
        )
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
                
        self.image_mean = []

    def forward(self):
        pass
