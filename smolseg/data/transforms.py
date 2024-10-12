from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Normalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.transform = transforms.Normalize(mean, std)

    def __call__(self, sample):
        images, masks = sample['image'], sample['mask']
        sample["orig_image"] = images
        images = self.transform(images)
        return {'image': images, 'mask': masks}
    

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
        images, masks = sample['image'], sample['mask']
        images = self.transform(images)
        return {'image': images, 'mask': masks}


class Resize:
    def __init__(self, size: Tuple[int, int] = (512, 1024)):
        self.size = size

    def __call__(self, sample):
        images, masks = sample['image'], sample['mask']
        images = nn.functional.interpolate(images.view(1, *images.shape), self.size, mode='bilinear')
        masks = nn.functional.interpolate(masks.view(1, 1, *masks.shape), self.size, mode='nearest')
        images, masks = images.view(*images.shape[1:]), masks.view(*masks.shape[2:])
        return {'image': images, 'mask': masks}

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
        return {"image": images, "mask": masks}


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        images, masks = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            images = torch.flip(images, [-1])
            masks = torch.flip(masks, [-1])
        return {"image": images, "mask": masks}


class PhotoMetricDistortion:
    def __init__(self, p=0.5):
        self.p = p
        self.transforms = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    def __call__(self, sample):
        images, masks = sample["image"], sample["mask"]
        if torch.rand(1) < self.p:
            images = self.transforms(images)
        return {"image": images, "mask": masks}