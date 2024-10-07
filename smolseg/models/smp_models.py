import torch.nn as nn
import segmentation_models_pytorch as smp

class SmpUnetWrapper(nn.Module):

    def __init__(self, model_name: str, num_classes: int, encoder_name: str = "resnet50", encoder_weights: str = "imagenet"):
        super().__init__()
        if model_name == "unet_seresnext26d":
            self.model = smp.Unet(
                encoder_name="tu-seresnext26d_32x4d",
                encoder_weights=encoder_weights,
                classes=num_classes
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def forward(self, x):
        return self.model(x)