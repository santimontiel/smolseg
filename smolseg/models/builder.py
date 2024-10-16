import torch.nn as nn
from smolseg.models.erfnet import ERFNet
from smolseg.models.smp_models import SmpUnetWrapper
from smolseg.models.torchvision_models import TorchvisionModelWrapper

def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    
    if model_name in [
        "unet_seresnext26d",
        "deeplabv3plus_seresnext26d",
        "deeplabv3plus_regnetz_b16",
        "deeplabv3plus_efficientnetv2_rw_t",
        "deeplabv3plus_efficientnetv2_rw_s",
    ]:
        if model_name.endswith("seresnext26d"):
            encoder_name = "tu-seresnext26d_32x4d"
        elif model_name.endswith("regnetz_b16"):
            encoder_name = "tu-regnetz_b16"
        elif model_name.endswith("efficientnetv2_rw_t"):
            encoder_name = "tu-efficientnetv2_rw_t"
        elif model_name.endswith("efficientnetv2_rw_s"):
            encoder_name = "tu-efficientnetv2_rw_s"
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
    elif model_name in ["erfnet"]:
        return ERFNet(num_classes=num_classes, in_channels=3)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")