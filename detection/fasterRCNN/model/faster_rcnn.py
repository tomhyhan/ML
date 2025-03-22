import torch
import torchvision
import torch.nn as nn


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
