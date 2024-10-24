import torch
from diffusers import StableDiffusionPipeline
import random
from PIL import Image  
from ..utils import Utils

class Txt2Img:

    @staticmethod
    def txt2img_post(prompt:str, guidance_scale:float = 0.7, num_inference_steps:int = 50, negative_prompt:str|list[str] = '', pipe:StableDiffusionPipeline = Utils.load_pipe(model_name="CompVis/stable-diffusion-v1-4")):
        '''
        txt2ImgPost (function) : create a post image

        parameters : 
        prompt (str) : sentence for the creation
        guidance_scale (float) : precision with the prompt
        num_inference_steps (int) : image quality
        negative_prompt (str) : object that you don't want
        '''
        seed = random.randint(1, 100)
        generator = torch.Generator(Utils.get_divice()).manual_seed(seed)   # type: ignore
        image = pipe(prompt, generator=generator, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]
        return image

    @staticmethod
    def txt2img_avatar(picture:Image, picture_type:["people", "animal", "background"], people_description:str, animal_description:str, background_description:str, artist:str, pipe:StableDiffusionPipeline = Utils.load_pipe(model_name="CompVis/stable-diffusion-v1-4")):
        '''
        txt2ImgAvatar (function) : create a avatar image
        
        parameters : 
        picture (str) : picture or painting image
        picture_type (str) : People Animal or Background picture
        people_description (list) : all infos on the person
        animal_description (list) : all infos on the animal
        background_description (str) : background prompt
        artist (str) : an artist for the creation
        '''
        prompt = ""
        match picture_type:
            case "people":
                prompt = f"A {people_description[0]} {people_description[1]} with {people_description[2]} hair, {people_description[3]} {people_description[4]} eyes, {people_description[5]}, {people_description[6]}, {people_description[7]}"
            case "animal":
                prompt = f"A {animal_description[0]} {animal_description[1]}"
            case "background":
                prompt = background_description
        
        if picture == "painting":
            prompt += f" by {artist}"
        
        seed = random.randint(1, 100)
        generator = torch.Generator(Utils.get_divice()).manual_seed(seed)
        image = pipe(prompt, generator=generator).images[0]
        return image

    @staticmethod
    def test()->Image:
        # Demander les inputs utilisateur
        prompt = str(input("Entrez votre création : "))
        guidance_scale = float(input("(Optionel) Ajoutez la précision du prompt : ").strip() or "7.5")
        num_inference_steps = int(input("(Optionel) Ajoutez la qualité de l'image : ").strip() or "50")
        negative_prompt = str(input("(Optionel) Ajoutez un élément à ne pas avoir sur l'image (negative_prompt) : "))

        # Générer et sauvegarder l'image
        return Txt2Img.txt2img_post(prompt, guidance_scale, num_inference_steps, negative_prompt)
