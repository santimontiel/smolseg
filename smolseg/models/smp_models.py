import torch.nn as nn
import segmentation_models_pytorch as smp

class SmpUnetWrapper(nn.Module):

    def __init__(self, model_name: str, num_classes: int, encoder_name: str = "resnet50", encoder_weights: str = "imagenet"):
        super().__init__()
        if model_name.startswith("unet"):
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=num_classes
            )
        elif model_name.startswith("deeplabv3plus"):
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=num_classes
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def forward(self, x):
        return self.model(x)