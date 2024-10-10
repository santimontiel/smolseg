import os
import sys
from typing import Callable, Dict
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import Tensor

from smolseg.models import build_model
from smolseg.module import SegmentationModule
from smolseg.data.cityscapes import Cityscapes

def load_pretrained_model(model_name: str, checkpoint_path: str) -> torch.nn.Module:
    
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

@torch.inference_mode()
def benchmark(model: Callable, inputs: Tensor) -> Dict[str, float]:

    # Warm up the model.
    model.eval()
    for _ in range(10):
        model(inputs)

    # Measure memory usage.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Time up to 10 runs.
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    for _ in range(100):
        model(inputs)
    end_time.record()
    torch.cuda.synchronize()

    return {
        "time": start_time.elapsed_time(end_time) / 100,             # ms
        "memory": torch.cuda.max_memory_allocated() / 1024**2,      # MB
        "params": sum(p.numel() for p in model.parameters()) / 1e6, # M params
    }


def main(
    model_name: str,
    checkpoint_path: str,
    batch_size: int = 1,
):
    model = load_pretrained_model(model_name, checkpoint_path).cuda()
    inputs = torch.randn(batch_size, 3, 512, 1024).cuda()
    results = benchmark(model, inputs)

    print(f"Benchmark results for {model_name}:")
    print(("-----------------------------------------------------"))
    print(f"Inference time:     {results['time']:.3f} ms")
    print(f"Throughput:         {1 / (results['time'] / 1000):.2f} FPS")
    print(f"Peak memory usage:  {results['memory']:.3f} MB")
    print(f"Model parameters:   {results['params']:.3f} M")


if __name__ == "__main__":
    main(
        model_name="deeplabv3plus_regnetz_b16",
        checkpoint_path="/workspace/logs/deeplabv3plus_regnetz_b16/checkpoints/best-epoch=epoch=170.ckpt",
    )