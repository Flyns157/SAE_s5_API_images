import torch
from diffusers import StableDiffusionPipeline
import random
from PIL import Image
from utils import Utils


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
    def txt2img_avatar(pipe: StableDiffusionPipeline):
        # make a generator for consistend results (good for testing)
        generator = torch.Generator("cuda").manual_seed(42)
        prompt_parts = []

        # Question 1:picture or painting?
        image_type = input(
            "Do you want a picture or a painting? ").strip().lower()
        if image_type == "painting":
            style = input("What style of painting? ").strip().lower()
            prompt_parts.append(f"a painting in {style} style ")
        else:
            prompt_parts.append(f"a portrait ")

        # Question 2: what subject do you want? (person, animale, landscape)
            subject = input(
                "What subject do you want? (person, animal, landscape) ").strip().lower()

        if subject == "person":
            gender = input("What is the gender? ").strip().lower()

            # Hair
            hair_color = input("What hair color? ").strip().lower()
            hair_length = input("What hair length? ").strip().lower()
            haircut = input("What haircut? ").strip().lower()

            # Nationality
            nationality = input("What nationality? ").strip().lower()

            # eyecolor
            eye_color = input("What eye color? ").strip().lower()

            # person prompt
            prompt_parts.append(
                f"of a {nationality} {gender} person with {eye_color} eyes and {hair_length}, {hair_color} hair (haircut: {haircut}).")

        elif subject == "animal":
            animal = input("what animal?").strip().lower()
            gender = input("What is the gender? ").strip().lower()
            body_color = input("What body color?").strip().lower()
            height = input("How big is it?").strip().lower()
            environment = input("Where is it?").strip().lower()

            # animal prompt
            prompt_parts.append(
                f"of a {height} {body_color} {animal} in a {environment}")

        elif subject == "landscape":
            # define favorites
            fav_color = input("Favorite color? ").strip().lower()
            fav_sport = input("Favorite sport? ").strip().lower()
            fav_animal = input("Favorite animal? ").strip().lower()
            fav_song = input("Favorite artist? ").strip().lower()
            fav_dish = input("Favorite dish? ").strip().lower()
            fav_job = input("current job? ").strip().lower()
            fav_hero = input("Favorite hero? ").strip().lower()

            prompt_parts.append(
                f"of a landscape with a {fav_color} theme, featuring {fav_sport}, a {fav_animal}, inspired by {fav_song}, and elements of {fav_dish}, {fav_job} and {fav_hero}")

        # Combine all parts in one prompt
        prompt = " ".join(prompt_parts)
        print(f"Generated prompt: {prompt}")
        # make image
        image = pipe(prompt, num_inference_steps=50,
                     generator=generator).images[0]

        return image

    @staticmethod
    def test():
        # Demander les inputs utilisateur
        prompt = str(input("Entrez votre création : "))
        guidance_scale = float(
            input("(Optionel) Ajoutez la précision du prompt : ").strip() or "7.5")
        num_inference_steps = int(
            input("(Optionel) Ajoutez la qualité de l'image : ").strip() or "50")
        negative_prompt = str(input(
            "(Optionel) Ajoutez un élément à ne pas avoir sur l'image (negative_prompt) : "))

        # Générer et sauvegarder l'image
        Txt2Img.txt2img_post(prompt, guidance_scale,
                             num_inference_steps, negative_prompt)
