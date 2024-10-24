from diffusers import StableDiffusionPipeline, DiffusionPipeline
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
        divice = 'cpu'
        if torch.cuda.is_available():
            divice = 'cuda'
        elif torch.backends.mps.is_available():
            divice = 'mps'
        return divice

    @staticmethod
    def load_pipe(model_name:str, loader:str = StableDiffusionPipeline, **kwargs)->DiffusionPipeline:
        '''
        Charge a Stable-Diffusion model in a pipeline
        '''
        device = Utils.get_divice()
        if device == "cuda":
            return loader.from_pretrained(model_name, torch_dtype=torch.float16, **kwargs).to(device)
        else:
            return loader.from_pretrained(model_name, **kwargs).to(device)
