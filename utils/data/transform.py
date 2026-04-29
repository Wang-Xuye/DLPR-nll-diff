import torch
import torchvision.transforms as T


import numpy as np


class PILToTensor(object):
    def __call__(self, pic):
        img = pic.view(pic.shape[1], pic.shape[0], 1)
        img = img.permute((2,0,1)).float()
        return img

class PILToTensor2(object):
    def __call__(self, pic):
        img = torch.as_tensor(pic.astype(np.float32))
        return img

 

def build_transforms(transform_type):
    
    if transform_type == "p64":
        transform = T.Compose([
            PILToTensor2(),
            T.RandomCrop(64),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            PILToTensor()
        ])
    elif transform_type == "p64_centercrop":
        transform = T.Compose([
            PILToTensor2(),
            T.CenterCrop(64),
            PILToTensor()
        ])

    else:
        raise Exception("No existing transform type {}.".format(transform_type))

    return transform
