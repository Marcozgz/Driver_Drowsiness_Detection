"""
Process the data.
"""
import glob
import random

import numpy as np
import cv2
import sys
import os
from PIL import Image
from torchvision import transforms

sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader


class WLFWDatasets(data.Dataset):
    def __init__(self, file_dir, transforms=None):
        self.transforms = transforms
        self.imgs_list = glob.glob(file_dir)
        # self.imgs_list = sorted(self.imgs_list)
        random.shuffle(self.imgs_list)

    def __getitem__(self, index):
        self.img = Image.open(self.imgs_list[index]).convert('RGB')
        self.img = self.img.resize((112, 112))
        if self.transforms:
            self.img = self.transforms(self.img)
        return self.img

    def __len__(self):
        return len(self.imgs_list)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    wlfwdataset = WLFWDatasets("/home/disk01/zgz/head_pose_estimation/data/drowsiness_dataset/test_data/all/*.png",
                               transform)
    dataloader = DataLoader(wlfwdataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    index = 0
    for img in dataloader:
        if index >= 500:
            break
        show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
        show_img = (show_img * 255).astype(np.uint8)
        np.clip(show_img, 0, 255)
        draw = show_img.copy()
        cv2.imwrite('./results/drowsiness_dataset/all/{}.png'.format(index), draw)
        index += 1
