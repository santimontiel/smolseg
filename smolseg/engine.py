import torch.nn as nn

from smolseg.models import build_model
from smolseg.module import SegmentationModule
from smolseg.data.cityscapes import Cityscapes

def load_pretrained_model(model_name: str, checkpoint_path: str) -> nn.Module:
    
    model = build_model(
        model_name=model_name,
        num_classes=Cityscapes.NUM_CLASSES,
        pretrained=False,
    )
    module = SegmentationModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        device="cuda",
        hparams={
            "lr": 3e-4,
            "weight_decay": 1e-3,
            "num_classes": Cityscapes.NUM_CLASSES,
            "max_epochs": 10,
        },
        train_dataloader=None,
    )
    return module.model