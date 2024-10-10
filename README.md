# smolseg
🌉 Smol (&lt;25M param) models for Real Time Semantic Segmentation

<div align=center>
    <img src=https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C.svg?style=for-the-badge&logo=pytorch>
    <img src=https://img.shields.io/badge/Lightning-2.4.0-purple?style=for-the-badge&logo=lightning>
    <img src=https://img.shields.io/badge/Wandb-0.18.3-yellow?style=for-the-badge&logo=weightsandbiases>
    <img src=https://img.shields.io/badge/Docker-%23007FFF?style=for-the-badge&logo=docker&logoColor=white&labelColor=%23007FFF>
</div>

## Model Zoo

| Model | Params | mIoU | Acc | FPS | Weights |
|-----|-----|-----|-----|-----|-----|
| DeepLabV3+ regnetz_b16 (smp)              | 10.7 M | 75.84 | 84.97 | 177.01 | --- |
| DeepLabV3 MobileNetV3 Large (torchvision) | 11.0 M | 68.61 | 79.59 | --- | --- |
| DeepLabV3+ seresnext26d (smp)             | 17.9 M | 74.05 | 82.71 | --- | --- |
| UNet seresnext26d (smp)                   | 23.8 M | 73.41 | 82.34 | --- | --- |

> [!NOTE]
> Frames per second (FPS) are measured in an NVIDIA RTX 4090 using *float32* precision.
