from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import string
import random
import base64
import torch
import io

from .database import Database

__all__ = ['Database', 'Utils']

class Utils:
    """
    Utility class for common utility functions.
    """
    
    @classmethod
    def generate_verification_code(cls, size: int = 6) -> str:
        """
        Generate a verification code.

        Parameters:
        size (int): Length of the verification code.

        Returns:
        str: The generated verification code.
        """
        CHARS = string.ascii_letters + string.digits
        return ''.join(random.choice(CHARS) for _ in range(size))
    
    @classmethod
    def get_divice(cls)->str:
        divice = 'cpu'
        if torch.cuda.is_available():
            divice = 'cuda'
        elif torch.backends.mps.is_available():
            divice = 'mps'
        return divice

    @classmethod
    def load_pipe(cls, model_name:str, loader:str = StableDiffusionPipeline, **kwargs)->DiffusionPipeline:
        '''
        Charge a Stable-Diffusion model in a pipeline
        '''
        device = Utils.get_divice()
        if device == "cuda":
            return loader.from_pretrained(model_name, torch_dtype=torch.float16, **kwargs).to(device)
        else:
            return loader.from_pretrained(model_name, **kwargs).to(device)

    @classmethod
    def decode_image(cls, image_b64):
        '''Helper function to decode a base64-encoded image'''
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        return image

    @classmethod
    def encode_image(cls, image):
        '''Helper function to encode a PIL image to base64'''
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return image_b64
