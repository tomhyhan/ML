import os
import csv
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
def extract_image(save_dir, csv_fname):
    with open(csv_fname) as f:
        reader = csv.reader(f)
        for idx, row in tqdm(enumerate(reader)):
            if idx == 0:
                continue
            im = np.zeros((784))
            im[:] = list(map(int, row[1:]))
            im = im.reshape((28,28))
            if not os.path.exists(os.path.join(save_dir, row[0])):
                os.mkdir(os.path.join(save_dir, row[0]))
            cv2.imwrite(os.path.join(os.path.join(save_dir, row[0]), f"{idx}.png"), im)
            
            if idx % 1000 == 0:
                print(f"finished created {idx} images in {save_dir}")

if __name__ == "__main__":
    extract_image("data/mnist/train/images", "data/mnist_train.csv")
    extract_image("data/mnist/test/images", "data/mnist_test.csv")