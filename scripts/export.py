import os
import sys
from PIL import Image
from typing import List
from urllib.request import urlopen
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxsim import simplify

from smolseg.engine import load_pretrained_model

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)

class ImagePreprocessor(nn.Module):

    def __init__(
        self,
        input_size: List[int],
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        x = nn.functional.interpolate(
            x, size=self.input_size[2:], mode="bilinear", align_corners=False
        )
        return x
    

def export_image_preprocessor() -> None:
    input_size = [1, 3, 512, 1024]
    output_onnx_file = "preprocessing.onnx"
    model = ImagePreprocessor(input_size=input_size)

    torch.onnx.export(
        model,
        torch.randn(input_size),
        output_onnx_file,
        opset_version=20,
        input_names=["input_rgb"],
        output_names=["output_preprocessing"],
        dynamic_axes={
            "input_rgb": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
        },
    )

    model_onnx = onnx.load(output_onnx_file)
    model_simplified, _ = simplify(model_onnx)
    onnx.save(model_simplified, output_onnx_file)


def export_model() -> None:
    model = load_pretrained_model(
        model_name="deeplabv3plus_regnetz_b16",
        checkpoint_path="/workspace/logs/deeplabv3plus_regnetz_b16/checkpoints/best-epoch=epoch=170.ckpt",
    ).eval().to("cpu")
    input_size = [1, 3, 512, 1024]
    output_onnx_file = "model.onnx"

    torch.onnx.export(
        model,
        torch.randn(input_size),
        output_onnx_file,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def merge_preprocessor_and_model() -> None:
    # Load the models
    model1 = onnx.load("preprocessing.onnx")
    model2 = onnx.load("model.onnx")

    # Merge the models
    merged_model = onnx.compose.merge_models(
        model1,
        model2,
        io_map=[("output_preprocessing", "input")],
        prefix1="preprocessing_",
        prefix2="model_",
        doc_string="Merged preprocessing and segmentation model",
    )

    # Save the merged model
    onnx.save(merged_model, "merged_model_compose.onnx")


def read_image(image: Image.Image):
    image = image.convert("RGB")
    img_numpy = np.array(image).astype(np.float32)
    img_numpy = img_numpy.transpose(2, 0, 1)
    img_numpy = np.expand_dims(img_numpy, axis=0)
    return img_numpy


def inference() -> None:

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_max_workspace_size": 8589934592,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_force_sequential_engine_build": False,
                "trt_max_partition_iterations": 10000,
                "trt_min_subgraph_size": 1,
                "trt_builder_optimization_level": 5,
                "trt_timing_cache_enable": True,
            },
        ),
    ]
    session = ort.InferenceSession("merged_model_compose.onnx", providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run([output_name], {input_name: read_image(img)})
    output = torch.from_numpy(output[0])
    print(output.shape)
    


def main() -> None:
    export_image_preprocessor()
    export_model()
    merge_preprocessor_and_model()
    inference()


if __name__ == "__main__":
    main()
