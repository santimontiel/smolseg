import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, List, Callable
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from smolseg.data.transforms import (
    Normalize,
    RandomResizeAndCrop,
    RandomFlip,
    Resize,
    ColorJitter,
    RandomValueAdd,
    RandomValueScale,
    RandomChannelSwap,
    GridMask,
)

@dataclass
class CityscapesClass:
    name: str
    id: int
    train_id: int
    category: str
    category_id: int
    has_instances: int
    ignore_in_eval: int
    color: Tuple[int, int, int]


class Cityscapes(Dataset):

    CLASSES = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]
    TRAIN_ID_TO_COLOR = [cls.color for cls in CLASSES if (cls.train_id != 255 and cls.train_id != -1)]
    TRAIN_ID_TO_COLOR.append((0, 0, 0))
    TRAIN_ID_TO_COLOR = np.array(TRAIN_ID_TO_COLOR)
    ID_TO_TRAIN_ID = [cls.train_id for cls in CLASSES]
    NUM_CLASSES = 19

    default_train_transforms = [
        RandomResizeAndCrop(),
        RandomFlip(),
        Resize(size=(512, 1024)),
        ColorJitter(),
        RandomValueAdd(),
        RandomValueScale(),
        RandomChannelSwap(),
        GridMask(0.2),
        Normalize(),
    ]
    default_val_transforms = default_test_transforms = [
        Resize(size=(512, 1024)),
        Normalize(),
    ]

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val"],
        transforms: Optional[List[Callable]] = None,
    ) -> None:
        
        # Ensure args are correct.
        if not os.path.isdir(root_dir):
            raise ValueError(f"root_dir must be a valid directory, not {root_dir}")
        if split not in ["train", "val"]:
            raise ValueError(f"split must be either 'train' or 'val', not {split}")
        
        # Save the arguments.
        self.root_dir = root_dir
        self.split = split
        self.mode = "gtFine"
        self.transforms = transforms

        # Set the default transforms.
        if self.transforms is None:
            self.transforms = self._set_default_transforms()

        # Locate the images and labels directories.
        self.images_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.labels_dir = os.path.join(root_dir, "gtFine", split)

        # Get the list of image and labels files.
        self.images, self.labels = [], []
        for city in os.listdir(self.images_dir):
            images_dir = os.path.join(self.images_dir, city)
            labels_dir = os.path.join(self.labels_dir, city)
            for image in os.listdir(images_dir):
                label = image.replace("leftImg8bit", "gtFine_labelIds")
                label = os.path.join(labels_dir, label)
                image = os.path.join(images_dir, image)
                self.images.append(image)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)
    
    def _set_default_transforms(self):
        if self.split == "train":
            return self.default_train_transforms
        elif self.split == "val":
            return self.default_val_transforms
        elif self.split == "test":
            return self.default_test_transforms
        else:
            raise ValueError(f"split must be either 'train', 'val', or 'test', not {self.split}")

    @staticmethod
    def _load_image(image_path: str) -> Tensor:
        image = read_image(image_path).float() / 255.0
        return image
    
    def _load_labels(self, label_path: str) -> Tensor:
        labels = np.array(Image.open(label_path))
        labels = np.array([self.ID_TO_TRAIN_ID[label] for label in labels.flatten()], dtype=np.uint8)
        labels = labels.reshape((1024, 2048))
        labels = torch.from_numpy(labels)
        return labels
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:

        data = {}
        image_path = self.images[index]
        label_path = self.labels[index]
        data["image"] = self._load_image(image_path)
        data["mask"] = self._load_labels(label_path)

        for transform in self.transforms:
            data = transform(data)
        
        data["mask"] = data["mask"].long()
        return data


