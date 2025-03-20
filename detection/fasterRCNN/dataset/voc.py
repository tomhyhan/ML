import glob
import random
from os import path
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

def load_images_and_anns(img_dir, ann_dir, label2idx):
    im_infos = []
    for ann_file in tqdm(glob.glob(path.join(ann_dir, "*.xml"))):
        im_info = {}
        im_info["img_id"] = path.basename(ann_file).split('.')[0]
        im_info["filename"] = path.join(img_dir, f"{im_info['img_id']}.jpg")
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        # get size width height and save in info
        size = root.find("size")
        im_info["width"] = int(size.find("width").text)
        im_info["height"] = int(size.find("height").text )       

        detections = []
        for obj in root.findall("object"):
            det = {}
            label = label2idx[obj.find("name").text]
            bndbox = obj.find("bndbox")
            bbox = [
                int(float(bndbox.find("xmin").text)) - 1,
                int(float(bndbox.find("ymin").text)) - 1,
                int(float(bndbox.find("xmax").text)) - 1,
                int(float(bndbox.find("ymax").text)) - 1
            ]
            det["label"] = label
            det["bbox"] = bbox
            detections.append(det)
        im_info["detections"] = detections
        im_infos.append(im_info)
    print(f"total {len(im_infos)} images found")
    return im_infos

class VOCDataset(Dataset):
    def __init__(self, split, img_dir ,ann_dir):
        self.split = split
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        
        classes.sort()
        classes += ["background"]
        self.label2idx = {c:i for i, c in enumerate(classes)}
        self.idx2label = {i:c for i, c in enumerate(classes)}
        self.images_info = load_images_and_anns(img_dir, ann_dir, self.label2idx)
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info["filename"])
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im_tensor = torchvision.transforms.ToTensor()(im)
        target = {}
        target["labels"] = torch.as_tensor([detection["label"] for detection in im_info["detections"]])
        target["bboxes"] = torch.as_tensor([detection["bbox"] for detection in im_info["detections"]])
        if to_flip:
            for idx, bbox in enumerate(target["bboxes"]):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                target["bboxes"][idx] = torch.tensor([x1, y1, x2, y2])
        return im_tensor, target, im_info["filename"]
        

if __name__ == "__main__":
    voc = VOCDataset("train", "VOC2007/JPEGImages", "VOC2007/Annotations")
    print(voc[0])