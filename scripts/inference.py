import os
import sys
from typing import Callable, List
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from smolseg.engine import load_pretrained_model
from smolseg.data.cityscapes import Cityscapes

DATA_DIR = "/data/cityscapes/leftImg8bit_sequence/test/munich/"

def colorize_argmax(argmax_image, color_map):
    # Ensure argmax_image is 2D
    assert argmax_image.ndim == 2, "Input image should be 2D"
    
    # Ensure color_map is 2D with 3 columns (RGB)
    assert color_map.ndim == 2 and color_map.shape[1] == 3, "Color map should be 2D with 3 columns"
    
    # Create the colorized image by using argmax values as indices into color_map
    colorized = color_map[argmax_image]
    
    return colorized.astype(np.uint8)


def create_video(images, masks, output_path, fps):
    # Get the height and width of the images
    height, width = masks[0].shape[:2]
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, 2 * height))
    
    # Write each image to the video
    for image, mask in zip(images, masks):
        frame = np.zeros((2 * height, width, 3), dtype=np.uint8)
        frame[:height] = image.permute(1, 2, 0).numpy() * 255
        frame[height:] = mask
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Release the VideoWriter object
    out.release()


def inference(model: Callable, inputs: List[str], device: str = "cuda") -> List[torch.Tensor]:

    model.eval().to(device)

    orig_images, outputs = [], []
    for input_path in tqdm(inputs, desc="Doing inference through images..."):

        image = Cityscapes._load_image(image_path=input_path)
        sample = {
            "image": image,
            "mask": torch.zeros((image.shape[:2]))
        }
        for transform in Cityscapes.default_test_transforms:
            image = transform(sample)
        image = sample["image"].unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs.append(model(image).detach().cpu())
        orig_images.append(sample["orig_image"].detach().cpu())

    return orig_images, outputs


def main(
    model_name: str,
    checkpoint_path: str,
    data_dir: str,
    device: str = "cuda",
    length_limit: int = 250,
):
    model = load_pretrained_model(model_name, checkpoint_path)
    inputs = sorted([os.path.join(data_dir, image) for image in os.listdir(data_dir)])[:length_limit]
    orig_images, outputs = inference(model, inputs, device)
    print(f"[INFER] Inference done succesfully through {len(outputs)} images")

    color_map = Cityscapes.TRAIN_ID_TO_COLOR
    colorized_outputs = [colorize_argmax(output.argmax(dim=1).cpu().squeeze().numpy(), color_map) for output in outputs]
    print(f"[COLOR] Colorization done succesfully through {len(colorized_outputs)} images")

    output_path = f"{model_name}.avi"
    create_video(orig_images, colorized_outputs, output_path, fps=10)
    print(f"Video succesfully saved to {output_path}!")
        

if __name__ == "__main__":
    main(
        model_name="deeplabv3plus_regnetz_b16",
        checkpoint_path="/workspace/logs/deeplabv3plus_regnetz_b16/checkpoints/best-epoch=epoch=170.ckpt",
        data_dir=DATA_DIR,
        device="cuda",
    )