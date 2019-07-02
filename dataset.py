from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import os, random
import PIL
from PIL import Image
import torchvision.transforms.functional as F

class LaminaDataset(Dataset):
    def __init__(self, prefix, transform=None, batch_nb=None):
        self.prefix_img = os.path.join(prefix, "img")
        self.prefix_gt = os.path.join(prefix, "gt")
        assert len(os.listdir(self.prefix_img))==len(os.listdir(self.prefix_gt))
        self.transform = transform
        self.batch_nb = batch_nb
    def __len__(self):
        if self.batch_nb is None:
            return len(os.listdir(self.prefix_img))
        else:
            return self.batch_nb
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.prefix_img, "%d.png" % idx))
        gt = Image.open(os.path.join(self.prefix_gt, "%d.png" % idx))
        img_gt = {'img': img, 'gt': gt}
        if self.transform is not None:
            img_gt = self.transform(img_gt)
        
        return img_gt
class ColorJitter(transforms.ColorJitter):
    def __call__(self, img_gt):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return {'img': transform(img_gt['img']), 'gt': img_gt['gt']}
class ToTensor(transforms.ToTensor):
    def __call__(self, img_gt):
        return {'img': F.to_tensor(img_gt['img']), 'gt': F.to_tensor(img_gt['gt'])}
class RandomAffine(transforms.RandomAffine):
    def __call__(self, img_gt):
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_gt['img'].size)
        img = F.affine(img_gt['img'], *ret, resample=PIL.Image.BILINEAR, fillcolor=self.fillcolor)
        gt = F.affine(img_gt['gt'], *ret, resample=PIL.Image.NEAREST, fillcolor=self.fillcolor)
        return {'img': img, 'gt': gt}
class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img_gt):
        i, j, h, w = self.get_params(img_gt['img'], self.scale, self.ratio)
        img = F.resized_crop(img_gt['img'], i, j, h, w, self.size, PIL.Image.BILINEAR)
        gt = F.resized_crop(img_gt['gt'], i, j, h, w, self.size, PIL.Image.NEAREST)
        return {'img': img, 'gt': gt}
class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img_gt):
        if random.random() < self.p:
            return {'img': F.vflip(img_gt['img']), 'gt': F.vflip(img_gt['gt'])}
        return img_gt
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img_gt):
        if random.random() < self.p:
            return {'img': F.hflip(img_gt['img']), 'gt': F.hflip(img_gt['gt'])}
        return img_gt