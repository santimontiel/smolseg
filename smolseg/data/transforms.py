from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf


class Normalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.transform = transforms.Normalize(mean, std)

    def __call__(self, sample):
        images = sample['image']
        sample["orig_image"] = images
        sample["image"] = self.transform(images)
        return sample
    

class Denormalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.transform = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )

    def __call__(self, sample):
        sample["images"] = self.transform(sample['image'])
        return sample


class Resize:
    def __init__(self, size: Tuple[int, int] = (512, 1024)):
        self.size = size

    def __call__(self, sample):
        images, masks = sample['image'], sample['mask']
        images = nn.functional.interpolate(images.view(1, *images.shape), self.size, mode='bilinear')
        masks = nn.functional.interpolate(masks.view(1, 1, *masks.shape), self.size, mode='nearest')
        images, masks = images.view(*images.shape[1:]), masks.view(*masks.shape[2:])
        sample["image"] = images
        sample["mask"] = masks
        return sample


class RandomResizeAndCrop:
    def __init__(
        self,
        original_size: Tuple[int, int] = (1024, 2048),
        ratio_scale: Tuple[float, float] = (0.5, 2.0),
        crop_size: Tuple[int, int] = (512, 1024),
    ):
        self.original_size = original_size
        self.ratio_scale = ratio_scale
        self.crop_size = crop_size

    def __call__(self, sample):
        images, masks = sample['image'], sample['mask']
        h, w = images.size()[-2:]

        # Random resize
        ratio = torch.empty(1).uniform_(*self.ratio_scale)
        new_h, new_w = int(h * ratio), int(w * ratio)
        images = nn.functional.interpolate(images.view(1, *images.shape), (new_h, new_w), mode="bilinear")
        masks = nn.functional.interpolate(masks.view(1, 1, *masks.shape), (new_h, new_w), mode="nearest")
        images, masks = images.view(*images.shape[1:]), masks.view(*masks.shape[2:])

        # Random crop
        i, j = (
            torch.randint(0, new_h - self.crop_size[0] + 1, (1,)).item(),
            torch.randint(0, new_w - self.crop_size[1] + 1, (1,)).item(),
        )
        images = images[..., i : i + self.crop_size[0], j : j + self.crop_size[1]]
        masks = masks[..., i : i + self.crop_size[0], j : j + self.crop_size[1]]
        sample["image"] = images
        sample["mask"] = masks
        return sample

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        images, masks = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            images = torch.flip(images, [-1])
            masks = torch.flip(masks, [-1])
        sample["image"] = images
        sample["mask"] = masks
        return sample

class ColorJitter:
    def __init__(self, p=0.5):
        self.p = p
        self.transforms = {
            "brightness": tf.adjust_brightness,
            "contrast": tf.adjust_contrast,
            "saturation": tf.adjust_saturation,
            "hue": tf.adjust_hue,
        }

    def __call__(self, sample):
        images = sample["image"]
        if torch.rand(1) < self.p:
            for name, transform in self.transforms.items():
                p = torch.rand(1)
                if p < 0.5:
                    p_factor = torch.rand(1)
                    p_factor = p_factor - 0.5 if name == "hue" else p_factor
                    images = transform(images, p_factor)
        sample["image"] = images
        return sample


class RandomChannelSwap:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        images = sample["image"]
        if torch.rand(1) < self.p:
            images = images[torch.randperm(3)]
        sample["image"] = images
        return sample
    

class RandomValueAdd:
    def __init__(
        self,
        p: float = 0.5,
        add_limits: Tuple[float, float] = (-32, 32),
    ) -> None:
        self.p = p
        self.add_limits = add_limits
    
    def __call__(self, sample):
        images = sample["image"]
        if torch.rand(1) < self.p:
            value = torch.rand(1).item() * (self.add_limits[1] - self.add_limits[0]) + self.add_limits[0]
            value /= 255.0
            images = images + value
        sample["image"] = images
        return sample
    

class RandomValueScale:
    def __init__(
        self,
        p: float = 0.5,
        scale_limits: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        self.p = p
        self.scale_limits = scale_limits

    def __call__(self, sample):
        images = sample["image"]
        if torch.rand(1) < self.p:
            scale = torch.rand(1).item() * (self.scale_limits[1] - self.scale_limits[0]) + self.scale_limits[0]
            images = images * scale
        sample["image"] = images
        return sample


class GridMask:
    def __init__(
        self,
        p: float = 0.5,
        num_grid: Tuple[int, int] = (8, 20),
        fill_value: float = 0.0,
    ) -> None:
        self.p = p
        self.num_grid = num_grid
        self.fill_value = fill_value

    def __call__(self, sample):
        images = sample["image"]
        if torch.rand(1) < self.p:
            h, w = images.size()[-2:]
            mask = torch.zeros(1, h, w)
            num_grid_h = torch.randint(*self.num_grid, (1,)).item()
            num_grid_w = torch.randint(*self.num_grid, (1,)).item()
            cell_h = h // num_grid_h
            cell_w = w // num_grid_w
            for i in range(num_grid_h):
                for j in range(num_grid_w):
                    if (i + j) % 2 == 0:
                        mask[..., i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = 1.0
            mask = mask.repeat(3, 1, 1)
            images = images * mask
        sample["image"] = images
        return sample

class StoreOriginalImage:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, sample):
        image = sample["image"].clone()
        return {"image": image, "mask": sample["mask"], "orig_image": image}