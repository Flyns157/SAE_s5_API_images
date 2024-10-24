import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from transformers import (
    CLIPTextModel, CLIPSegProcessor, CLIPSegForImageSegmentation
)
from diffusers import UniPCMultistepScheduler
from safetensors.torch import load_model
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
    StableDiffusionPowerPaintBrushNetPipeline
)
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.utils.utils import TokenizerWrapper, add_tokens

# Ensure the correct path for importing local modules
sys.path.insert(0, os.path.join(os.getcwd(), "../generator/content/PowerPaint"))

# Constants and configurations
CHECKPOINT_DIR = "/content/checkpoints"
BASE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "realisticVisionV60B1_v51VAE")
LOCAL_FILES_ONLY = True
DEFAULT_NEGATIVE_PROMPT = (
    "out of frame, lowres, error, cropped, worst quality, low quality, "
    "jpeg artifacts, ugly, duplicate, morbid, mutilated, deformed, blurry, "
    "dehydrated, bad anatomy, bad proportions, extra limbs, malformed limbs, "
    "watermark, signature"
)

# Initialize models
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
segmentation_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16
)
text_encoder_brushnet = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", torch_dtype=torch.float16
)
brushnet = BrushNetModel.from_unet(unet)

pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
    BASE_MODEL_PATH,
    brushnet=brushnet,
    text_encoder_brushnet=text_encoder_brushnet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_PATH, subfolder="unet", torch_dtype=torch.float16, local_files_only=LOCAL_FILES_ONLY
)
pipe.tokenizer = TokenizerWrapper(
    from_pretrained=BASE_MODEL_PATH, subfolder="tokenizer", torch_type=torch.float16, local_files_only=LOCAL_FILES_ONLY
)

# Add learned task tokens to the tokenizer
add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder_brushnet,
    placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
    initialize_tokens=["a"] * 3,
    num_vectors_per_token=10,
)

load_model(brushnet, os.path.join(CHECKPOINT_DIR, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"))
pipe.text_encoder_brushnet.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe = pipe.to("cpu")

# Helper functions
def task_to_prompt(task: str) -> tuple[str, str, str, str]:
    task_mapping = {
        "object-removal": ("P_ctxt", "P_ctxt", "P_obj", "P_obj"),
        "context-aware": ("P_ctxt", "P_ctxt", "", ""),
        "shape-guided": ("P_shape", "P_ctxt", "P_shape", "P_ctxt"),
        "image-outpainting": ("P_ctxt", "P_ctxt", "P_obj", "P_obj"),
        "default": ("P_obj", "P_obj", "P_obj", "P_obj")
    }
    return task_mapping.get(task, task_mapping["default"])

@torch.inference_mode()
def predict(
    pipe, input_image: dict, prompt: str, fitting_degree: float, ddim_steps: int,
    scale: float, negative_prompt: str, task: str
) -> Image.Image:
    promptA, promptB, negative_promptA, negative_promptB = task_to_prompt(task)
    print(task, promptA, promptB, negative_promptA, negative_promptB)

    img = np.array(input_image["image"].convert("RGB"))
    W, H = (dim - dim % 8 for dim in img.shape[:2])

    # Resize images
    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))

    np_inpimg = np.array(input_image["image"]) * (1 - np.array(input_image["mask"]) / 255.0)
    input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

    return pipe(
        promptA=promptA, promptB=promptB, promptU=prompt,
        tradoff=fitting_degree, tradoff_nag=fitting_degree,
        image=input_image["image"].convert("RGB"), mask=input_image["mask"].convert("RGB"),
        num_inference_steps=ddim_steps, brushnet_conditioning_scale=1.0,
        negative_promptA=negative_promptA, negative_promptB=negative_promptB,
        negative_promptU=negative_prompt, guidance_scale=scale, width=H, height=W
    ).images[0]

# Task-specific functions
def object_removal(pipe, init_image: Image.Image, mask_image: Image.Image, negative_prompt: str, fitting_degree=1, num_inference_steps=50, guidance_scale=12) -> Image.Image:
    negative_prompt += f", {DEFAULT_NEGATIVE_PROMPT}"
    input_image = {"image": init_image, "mask": mask_image}
    return predict(pipe, input_image, "empty scene blur", fitting_degree, num_inference_steps, guidance_scale, negative_prompt, "object-removal")

def object_addition(pipe, init_image: Image.Image, mask_image: Image.Image, prompt: str,
                    fitting_degree=1, num_inference_steps=50, guidance_scale=12) -> Image.Image:
    input_image = {"image": init_image, "mask": mask_image}
    return predict(pipe, input_image, prompt, fitting_degree, num_inference_steps, guidance_scale, "", "text-guided")

def generate_mask_image(init_image: Image.Image, mask_prompt: str | list[str], debug=False) -> Image.Image:
    if isinstance(mask_prompt, str):
        mask_prompt = [mask_prompt, mask_prompt]
    elif len(mask_prompt) == 1:
        mask_prompt *= 2

    inputs = processor(text=mask_prompt, images=[init_image] * len(mask_prompt), return_tensors="pt")
    outputs = segmentation_model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    if debug:
        _, ax = plt.subplots(1, 5, figsize=(15, len(mask_prompt) + 1))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(init_image)
        [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(mask_prompt))]

    plt.imsave("temp.png", torch.sigmoid(preds[1][0]))
    img2 = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)
    _, bw_image = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)

    return Image.fromarray(bw_image)
