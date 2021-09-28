import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
import csv
from PIL import ImageFilter, ImageOps
from .base_datasets import BaseDataset

class GridDataset(BaseDataset):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

        self.forward_encoding = cfg.model.forward_encoding
        self.pyramid_list = self._parse_encoding_params()

        self.grid_ratio_default = 2.0
        self.phase = phase

    def _parse_encoding_params(self):
        idx = self.forward_encoding.find('-')
        if idx < 0:
            return []
        blocks = self.forward_encoding[idx + 1:].split(',')
        blocks = [int(s) for s in blocks]
        return blocks

    def get_grid_location(self, size, ratio, num_grid):
        '''
        :param size: size of the height/width
        :param ratio: generate grid size/ even divided grid size
        :param num_grid: number of grid
        :return: a list containing the coordinate of the grid
        '''
        raw_grid_size = int(size / num_grid)
        enlarged_grid_size = int(size / num_grid * ratio)

        center_location = raw_grid_size // 2

        location_list = []
        for i in range(num_grid):
            location_list.append((max(0, center_location - enlarged_grid_size // 2),
                                  min(size, center_location + enlarged_grid_size // 2)))
            center_location = center_location + raw_grid_size

        return location_list

    def get_pyramid(self, img, num_grid):
        if self.phase == 'train':
            grid_ratio = 1 + 2 * random.random()
        else:
            grid_ratio = self.grid_ratio_default
        w, h = img.size
        grid_locations_w = self.get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = self.get_grid_location(h, grid_ratio, num_grid)

        patches_list=[]
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w=grid_locations_w[j]
                patch_location_h=grid_locations_h[i]
                left_up_corner_w=patch_location_w[0]
                left_up_corner_h=patch_location_h[0]
                right_down_cornet_w=patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch=img.crop((left_up_corner_w,left_up_corner_h,right_down_cornet_w,right_down_cornet_h))
                patch=self.transform(patch)
                patches_list.append(patch)
        return patches_list

    def prepare_transform(self, cfg, phase):
        norm = transforms.Normalize(
            np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
        )
        if cfg.model.encoder == "FourLayer_64F":

            if phase == "train":
                t = [
                    transforms.Resize([84, 84]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]
        else:
            if phase == "train":
                t = [
                    transforms.Resize([84, 84]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]

        return transforms.Compose(t)

    def _get_griditems(self, img):
        pyramid_list = []
        for num_grid in self.pyramid_list:
            patches = self.get_pyramid(img, num_grid)
            pyramid_list.extend(patches)
        pyramid_list = torch.cat(pyramid_list, dim=0)
        return pyramid_list

    def __getitem__(self, index):
        episode = self.data_list[index]
        support_x, support_y, query_x, query_y = [], [], [], []
        for e in episode:
            query_ = e["query_x"]
            for q in query_:
                im = self._get_griditems(Image.open(q).convert("RGB"))

                query_x.append(im.unsqueeze(0))
            support_ = e["support_x"]
            for s in support_:
                im = self._get_griditems(Image.open(s).convert("RGB"))
                support_x.append(im.unsqueeze(0))
            target = e["target"]
            support_y.extend(np.tile(target, len(support_)))
            query_y.extend(np.tile(target, len(query_)))

        support_x = torch.cat(support_x, 0)
        query_x = torch.cat(query_x, 0)
        support_y = torch.LongTensor(support_y)
        query_y = torch.LongTensor(query_y)

        randperm = torch.randperm(len(query_y))
        query_x = query_x[randperm]
        query_y = query_y[randperm]

        return support_x, support_y, query_x, query_y

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class AddPepperNoise(object):

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255
            img_[mask == 2] = 0
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img