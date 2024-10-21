import sys
import os
sys.path.insert(0,os.path.join("./PowerPaint"))

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
pipe = pipe.to("cuda")