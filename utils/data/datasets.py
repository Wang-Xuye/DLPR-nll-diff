import torch
from torch.utils import data

import os
import numpy as np



class ImageDataset(data.Dataset):

    def __init__(self, path_dir, img_mode = None, transform=None):
        self.path_dir = path_dir
        self.img_mode = img_mode
        self.transform = transform
        self.images = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        img_path = os.path.join(self.path_dir, image_name)
        img = np.load(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img

