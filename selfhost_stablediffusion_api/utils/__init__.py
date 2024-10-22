from diffusers import StableDiffusionPipeline
import string
import random
import torch

from .database import Database

__all__ = ['Database', 'Utils']

class Utils:
    """
    Utility class for common utility functions.
    """
    
    @staticmethod
    def generate_verification_code(size: int = 6) -> str:
        """
        Generate a verification code.

        Parameters:
        size (int): Length of the verification code.

        Returns:
        str: The generated verification code.
        """
        CHARS = string.ascii_letters + string.digits
        return ''.join(random.choice(CHARS) for _ in range(size))
    
    @staticmethod
    def get_divice()->str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def load_SD_pipe(name:str)->StableDiffusionPipeline:
        '''
        Charge a Stable-Diffusion model in a pipeline
        '''
        device = Utils.get_divice()
        if device == "cuda":
            return StableDiffusionPipeline.from_pretrained(name, torch_dtype=torch.float16).to(device)
        else:
            return StableDiffusionPipeline.from_pretrained(name).to(device)
