from diffusers import StableDiffusionImg2ImgPipeline
from ..utils import Utils
from PIL import Image
import torch

class Img2Img:
    
    @staticmethod
    def img2img(prompt, init_image, strength, num_inference_steps, pipe:StableDiffusionImg2ImgPipeline = Utils.load_pipe(model_name="CompVis/stable-diffusion-v1-4", loader=StableDiffusionImg2ImgPipeline))->Image:
        return pipe(rompt = prompt,
                    image = init_image, 
                    generator = torch.Generator(Utils.get_divice()).manual_seed(123456),
                    strength = strength, # Between 0 and 1 (1 for maximum changes),
                    num_inference_steps = num_inference_steps, # Defaults to 50 for good results
                    ).images[0]

    @staticmethod
    def test()->tuple[Image, Image]:
        device_available = Utils.get_divice()
        img2img_pipe = Utils.load_pipe(model_name="stabilityai/stable-diffusion-2-1-base", loader=StableDiffusionImg2ImgPipeline, variant="fp16", use_safetensors=True).to(device_available)
        prompt = "An avatar of a man. Add a sword in his hand"

        from .txt2Img import Txt2Img
        init_image = Txt2Img.txt2img_post(prompt=prompt)

        return init_image, Img2Img.img2img(prompt=prompt, init_image=init_image, strength=1, num_inference_steps=50, pipe=img2img_pipe)
