import os
import albumentations as albu
import cv2
import torch
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET


def load_images_and_anns(im_sets, label2idx, ann_fname, split):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_sets: Sets of images to consider
    :param label2idx: Class Name to index mapping for dataset
    :param ann_fname: txt file containing image names{trainval.txt/test.txt}
    :param split: train/test
    :return:
    """
    im_infos = []
    ims = []

    for im_set in im_sets:
        im_names = []
        # Fetch all image names in txt file for this imageset
        for line in open(os.path.join(
                im_set, 'ImageSets', 'Main', '{}.txt'.format(ann_fname))):
            im_names.append(line.strip())

        # Set annotation and image path
        ann_dir = os.path.join(im_set, 'Annotations')
        im_dir = os.path.join(im_set, 'JPEGImages')
        for im_name in im_names:
            ann_file = os.path.join(ann_dir, '{}.xml'.format(im_name))
            im_info = {}
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
            im_info['filename'] = os.path.join(
                im_dir, '{}.jpg'.format(im_info['img_id'])
            )
            im_info['width'] = width
            im_info['height'] = height
            detections = []

            # We will keep an image only if there are valid rois in it
            any_valid_object = False
            for obj in ann_info.findall('object'):
                det = {}
                label = label2idx[obj.find('name').text]
                difficult = int(obj.find('difficult').text)
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(float(bbox_info.find('xmin').text))-1,
                    int(float(bbox_info.find('ymin').text))-1,
                    int(float(bbox_info.find('xmax').text))-1,
                    int(float(bbox_info.find('ymax').text))-1
                ]
                det['label'] = label
                det['bbox'] = bbox
                det['difficult'] = difficult
                # Ignore difficult rois during training
                # At test time eval does the job of ignoring difficult
                # examples. 
                if difficult == 0 or split == 'test':
                    detections.append(det)
                    any_valid_object = True

            if any_valid_object:
                im_info['detections'] = detections
                im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


class VOCDataset(Dataset):
    def __init__(self, split, im_sets, im_size=448, S=7, B=2, C=20):
        self.split = split
        # Imagesets for this dataset instance (VOC2007/VOC2007+VOC2012/VOC2007-test)
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        self.im_size = im_size
        # Grid size, B and C parameter for target setting
        self.S = S
        self.B = B
        self.C = C

        # Train and test transformations
        self.transforms = {
            'train': albu.Compose([
                albu.HorizontalFlip(p=0.5),
                albu.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.2, 0.2),
                    always_apply=True
                ),
                albu.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.2, 0.2),
                    always_apply=None,
                    p=0.5,
                ),
                albu.Resize(self.im_size, self.im_size)],
                bbox_params=albu.BboxParams(format='pascal_voc',
                                            label_fields=['labels'])),
            'test': albu.Compose([
                albu.Resize(self.im_size, self.im_size),
                ],
                bbox_params=albu.BboxParams(format='pascal_voc',
                                            label_fields=['labels']))
        }

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(self.im_sets,
                                                self.label2idx,
                                                self.fname,
                                                self.split)
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = cv2.imread(im_info['filename'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        bboxes = [detection['bbox'] for detection in im_info['detections']]
        labels = [detection['label'] for detection in im_info['detections']]
        difficult = [detection['difficult'] for detection in im_info['detections']]

        # Transform Image and ann according to augmentations list
        transformed_info = self.transforms[self.split](image=im,
                                                       bboxes=bboxes,
                                                       labels=labels)
        im = transformed_info['image']
        bboxes = torch.as_tensor(transformed_info['bboxes'])
        labels = torch.as_tensor(transformed_info['labels'])
        difficult = torch.as_tensor(difficult)
        
        # conver image to tensor and normalize
        # r 0.485 / 0.229 
        # g 0.456 / 0.224
        # b 0.406 / 0.225
        im_tensor = torch.from_numpy(im / 255.).permute(1,2,0).float()
        r = (im_tensor[0].unsqueeze(0) - 0.485) / 0.229
        g = (im_tensor[1].unsqueeze(0) - 0.456) / 0.224
        b = (im_tensor[2].unsqueeze(0) - 0.406) / 0.225
        im_tensor = torch.cat([r,g,b], dim=0)
        bboxes_tensor = torch.as_tensor(bboxes)
        labels_tensor = torch.as_tensor(labels)
        
        # create target for YOLO with target dim
        target_dim = self.B * 5 + self.C
        yolo_target = torch.zeros(self.S, self.S, target_dim)
        
        # Width and Height of each grid cell H // S
        H, W = im_tensor.shape[-2:]
        cell_pixels = H / self.S
        
        # if there exist boxes
        if len(bboxes) > 0:
            # convert x1y1x2y2 to xywh
            w = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]
            h = bboxes_tensor[:, 3] - bboxes_tensor[:, 1]
            xc = bboxes_tensor[:,0] + 0.5 * w
            yc = bboxes_tensor[:,1] + 0.5 * h
            
            # get top-left corner cell i, j from xc yc
            box_i = torch.floor(xc / cell_pixels)
            box_j = torch.floor(yc / cell_pixels)
            
            # xc, yc offset from cell top-left
            xc_offset = (xc - box_i*cell_pixels) / cell_pixels
            yc_offset = (yc - box_j*cell_pixels) / cell_pixels
            
            # normalize box w and h
            box_w_label = w / W
            bex_h_label = h / H
            
            # fill in yolo targets from gt box we found
            for idx in range(len(bboxes)):
                for i in range(self.B):
                    s = 5 * i
                    yolo_target[box_j[idx], box_i[idx], s] = xc_offset[idx]
                    yolo_target[box_j[idx], box_i[idx], s+1] = yc_offset[idx]
                    yolo_target[box_j[idx], box_i[idx], s+2] = box_w_label[idx].sqrt()
                    yolo_target[box_j[idx], box_i[idx], s+3] = bex_h_label[idx].sqrt()
                    yolo_target[box_j[idx], box_i[idx], s+4] = 1.0
                label = int(labels[idx])
                classes = torch.zeros(self.C)
                classes[label] = 1
                yolo_target[box_j[idx], box_i[idx], self.B*5 + 10] = classes
        
        if len(bboxes) > 0:
            bboxes_tensor /= torch.tensor([W,H,W,H]).expand_as(bboxes_tensor)
            
        targets = {
            "bboxes": bboxes_tensor,
            "labels": labels_tensor,
            "yolo_targets": yolo_target,
            "difficult": difficult
        }      
        
        return im_tensor, targets, im_info["filename"]
            
