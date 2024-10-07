import torch.nn as nn
import torchvision.models.segmentation as models


class TorchvisionModelWrapper(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = False):
        super().__init__()
        if model_name == "deeplabv3_mobilenetv3_large":
            self.model = models.deeplabv3_mobilenet_v3_large(
                pretrained=False, num_classes=num_classes
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def forward(self, x):
        return self.model(x)["out"]
