import torch
from diffusers import StableDiffusionPipeline
import random
from peft import get_lora_model, LoraConfig, TaskType
from PIL import Image  
from ..utils import Utils


class Txt2Img:

    @staticmethod
    def txt2img_post(prompt: str, guidance_scale: float, num_inference_steps: int, negative_prompt: str | list[str], pipe: StableDiffusionPipeline = Utils.load_SD_pipe(name="CompVis/stable-diffusion-v1-4")):
        '''
        txt2ImgPost (function) : create a post image

        parameters : 
        prompt (str) : sentence for the creation
        guidance_scale (float) : precision with the prompt
        num_inference_steps (int) : image quality
        negative_prompt (str) : object that you don't want
        '''
        seed = random.randint(1, 100)
        generator = torch.Generator(
            Utils.get_divice()).manual_seed(seed)   # type: ignore
        image = pipe(prompt, generator=generator, guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]
        return image

    @staticmethod
    def txt2img_avatar(image_type: str, style: str = None, 
                        subject: str = None, gender: str = None, hair_color: str = None, 
                        hair_length: str = None, haircut: str = None, nationality: str = None, 
                        eye_color: str = None, animal: str = None, body_color: str = None, 
                        height: str = None, environment: str = None, fav_color: str = None, 
                        fav_sport: str = None, fav_animal: str = None, fav_song: str = None, 
                        fav_dish: str = None, fav_job: str = None, fav_hero: str = None,
                        pipe: StableDiffusionPipeline = Utils.load_pipe(model_name="CompVis/stable-diffusion-v1-4"), **kwargs):
        """
        Generates an image based on user responses to specific questions.

        Parameters:
        - image_type (str): "picture" or "painting" (Question 1: Do you want a picture or a painting?)
        - style (str, optional): Style of painting (if applicable) (Question 2: What style of painting?)
        - subject (str, optional): "person", "animal", or "landscape" (Question 3: What subject do you want?)
        - gender (str, optional): Gender of person or animal (Question 4: What is the gender?)
        - hair_color (str, optional): Hair color (Question 5: What hair color?)
        - hair_length (str, optional): Hair length (Question 6: What hair length?)
        - haircut (str, optional): Haircut style (Question 7: What haircut?)
        - nationality (str, optional): Nationality (Question 8: What nationality?)
        - eye_color (str, optional): Eye color (Question 9: What eye color?)
        - animal (str, optional): Type of animal (Question 10: What animal?)
        - body_color (str, optional): Body color of the animal (Question 11: What body color?)
        - height (str, optional): Size of the animal (Question 12: How big is it?)
        - environment (str, optional): Location of the animal (Question 13: Where is it?)
        - fav_color (str, optional): Favorite color for landscape (Question 14: Favorite color?)
        - fav_sport (str, optional): Favorite sport for landscape (Question 15: Favorite sport?)
        - fav_animal (str, optional): Favorite animal for landscape (Question 16: Favorite animal?)
        - fav_song (str, optional): Favorite artist/song for landscape (Question 17: Favorite artist?)
        - fav_dish (str, optional): Favorite dish for landscape (Question 18: Favorite dish?)
        - fav_job (str, optional): Current job for landscape (Question 19: Current job?)
        - fav_hero (str, optional): Favorite hero for landscape (Question 20: Favorite hero?)
        
        Returns:
        - An image generated based on the given prompt.
        """
        # make a generator for consistent results (good for testing)
        generator = torch.Generator("cuda").manual_seed(42)
        prompt_parts = []

        # Determine image type
        if image_type == "painting" and style:
            prompt_parts.append(f"a painting in {style} style ")
        else:
            prompt_parts.append(f"a portrait ")

        # Subject-based prompts
        if subject == "person":
            prompt_parts.append(
                f"of a {nationality} {gender} person with {eye_color} eyes and {hair_length}, {hair_color} hair (haircut: {haircut}).")
        elif subject == "animal":
            prompt_parts.append(
                f"of a {height} {body_color} {animal} in a {environment}")
        elif subject == "landscape":
            prompt_parts.append(
                f"of a landscape with a {fav_color} theme, featuring {fav_sport}, a {fav_animal}, inspired by {fav_song}, and elements of {fav_dish}, {fav_job}, and {fav_hero}")

        # Combine all parts into one prompt
        prompt = " ".join(prompt_parts)
        print(f"Generated prompt: {prompt}")

        # Generate image
        images = pipe(prompt, num_inference_steps=50, generator=generator).images
        return image[0]

    @staticmethod
    def test()->Image:
        # Demander les inputs utilisateur
        prompt = str(input("Entrez votre création : "))
        guidance_scale = float(
            input("(Optionel) Ajoutez la précision du prompt : ").strip() or "7.5")
        num_inference_steps = int(
            input("(Optionel) Ajoutez la qualité de l'image : ").strip() or "50")
        negative_prompt = str(input(
            "(Optionel) Ajoutez un élément à ne pas avoir sur l'image (negative_prompt) : "))

        # Générer et sauvegarder l'image
        return Txt2Img.txt2ImgPost(prompt, guidance_scale, num_inference_steps, negative_prompt)
