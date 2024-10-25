from .txt2img import Txt2Img
from .img2img import Img2Img
# from .inpainting_main import Inpainting
from .finetuningInpainting import ImageProcessor as FineInpainting

__all__ = ['Txt2Img', 'Img2Img', 'FineInpainting'] # , 'Inpainting'
