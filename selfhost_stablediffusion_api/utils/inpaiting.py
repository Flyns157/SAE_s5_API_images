import sys
import os

import PIL

sys.path.insert(0, os.path.join(os.getcwd(), "content/PowerPaint"))
print(os.path.join(os.getcwd(), "content/PowerPaint"))

from safetensors.torch import load_model, save_model
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler

from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
    StableDiffusionPowerPaintBrushNetPipeline,
)

#from powerpaint.power_paint_tokenizer import PowerPaintTokenizer
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.utils.utils import TokenizerWrapper, add_tokens
from diffusers import UniPCMultistepScheduler

checkpoint_dir = "/content/checkpoints"
local_files_only = True

# brushnet-based version
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="unet",
    revision=None,
    torch_dtype=torch.float16,
    local_files_only=False,
)
text_encoder_brushnet = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="text_encoder",
    revision=None,
    torch_dtype=torch.float16,
    local_files_only=False,
)
brushnet = BrushNetModel.from_unet(unet)
base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
    base_model_path,
    brushnet=brushnet,
    text_encoder_brushnet=text_encoder_brushnet,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    safety_checker=None,
)
pipe.unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
    revision=None,
    torch_dtype=torch.float16,
    local_files_only=local_files_only,
)
pipe.tokenizer = TokenizerWrapper(
    from_pretrained=base_model_path,
    subfolder="tokenizer",
    revision=None,
    torch_type=torch.float16,
    local_files_only=local_files_only,
)

# add learned task tokens into the tokenizer
add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder_brushnet,
    placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
    initialize_tokens=["a", "a", "a"],
    num_vectors_per_token=10,
)
load_model(
    pipe.brushnet,
    os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
)

pipe.text_encoder_brushnet.load_state_dict(
    torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
pipe = pipe.to("cpu")

def task_to_prompt(control_type):
    if control_type == "object-removal":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"
    elif control_type == "context-aware":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = ""
        negative_promptB = ""
    elif control_type == "shape-guided":
        promptA = "P_shape"
        promptB = "P_ctxt"
        negative_promptA = "P_shape"
        negative_promptB = "P_ctxt"
    elif control_type == "image-outpainting":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"
    else:
        promptA = "P_obj"
        promptB = "P_obj"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB

@torch.inference_mode()
def predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    scale,
    negative_prompt,
    task,
):
    promptA, promptB, negative_promptA, negative_promptB = task_to_prompt(task)
    print(task, promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))

    np_inpimg = np.array(input_image["image"])
    np_inmask = np.array(input_image["mask"]) / 255.0

    np_inpimg = np_inpimg * (1 - np_inmask)

    input_image["image"] = PIL.Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

    result = pipe(
        promptA=promptA,
        promptB=promptB,
        promptU=prompt,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        image=input_image["image"].convert("RGB"),
        mask=input_image["mask"].convert("RGB"),
        num_inference_steps=ddim_steps,
        brushnet_conditioning_scale=1.0,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        negative_promptU=negative_prompt,
        guidance_scale=scale,
        width=H,
        height=W,
    ).images[0]
    return result


def object_removal_with_instruct_inpainting(pipe, init_image, mask_image, negative_prompt, fitting_degree=1, \
                                          num_inference_steps=50, guidance_scale=12):
    negative_prompt = negative_prompt + ", out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
    input_image = {"image": init_image, "mask": mask_image}
    image = predict(
        pipe,
        input_image,
        "empty scene blur", # prompt
        fitting_degree,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        "object-removal" # task
    )
    return image

def object_addition_with_instruct_inpainting(pipe, init_image, mask_image, prompt, fitting_degree=1, \
                                          num_inference_steps=50, guidance_scale=12):
    input_image = {"image": init_image, "mask": mask_image}
    image = predict(
        pipe,
        input_image,
        prompt,
        fitting_degree,
        num_inference_steps,
        guidance_scale,
        "", # negative prompt
        "text-guided" # task
    )
    return image