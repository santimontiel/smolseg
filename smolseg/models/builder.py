import torch.nn as nn
from smolseg.models.smp_models import SmpUnetWrapper
from smolseg.models.torchvision_models import TorchvisionModelWrapper

def build_model(
    model_name: str,
    num_classes: int,
    encoder_name: str = "resnet50",
    pretrained: bool = False,
) -> nn.Module:
    
    if model_name in ["unet_seresnext26d"]:
        return SmpUnetWrapper(
            model_name=model_name,
            num_classes=num_classes,
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
        )
    elif model_name in ["deeplabv3_mobilenetv3_large"]:
        return TorchvisionModelWrapper(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")