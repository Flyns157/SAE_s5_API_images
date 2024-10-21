import torch
from diffusers import StableDiffusionPipeline
import random
from PIL import Image  

# charge model
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# choose the GPU
pipe = pipe.to(device)

def txt2ImgPost(prompt, guidance_scale, num_inference_steps, negative_prompt):
    '''
    txt2ImgPost (function) : create a post image

    parameters : 
    prompt (str) : sentence for the creation
    guidance_scale (float) : precision with the prompt
    num_inference_steps (int) : image quality
    negative_prompt (str) : object that you don't want
    '''
    seed = random.randint(1, 100)
    generator = torch.Generator(device).manual_seed(seed)  
    image = pipe(prompt, generator=generator, guidance_scale=guidance_scale, 
                 num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]
    return image

def txt2ImgAvatar(picture, typePicture, descriptionPeople, descriptionAnimal, descriptionBackground, artist):
    '''
    txt2ImgAvatar (function) : create a avatar image
    
    parameters : 
    picture (str) : picture or painting image
    typePicture (str) : People Animal or Background picture
    descriptionPeople (list) : all infos on the person
    descriptionAnimal (list) : all infos on the animal
    descriptionBackground (str) : background prompt
    artist (str) : an artist for the creation
    '''
    prompt = ""
    match typePicture:
        case "people":
            prompt = f"A {descriptionPeople[0]} {descriptionPeople[1]} with {descriptionPeople[2]} hair, {descriptionPeople[3]} {descriptionPeople[4]} eyes, {descriptionPeople[5]}, {descriptionPeople[6]}, {descriptionPeople[7]}"
        case "animal":
            prompt = f"A {descriptionAnimal[0]} {descriptionAnimal[1]}"
        case "background":
            prompt = descriptionBackground
        case _:
            pass
    
    if picture == "painting":
        prompt += f" by {artist}"
    
    seed = random.randint(1, 100)
    generator = torch.Generator(device).manual_seed(seed) 
    image = pipe(prompt, generator=generator).images[0]
    return image

# Demander les inputs utilisateur
prompt = str(input("Entrez votre création : "))
guidance_scale = float(input("(Optionel) Ajoutez la précision du prompt : ").strip() or "7.5")
num_inference_steps = int(input("(Optionel) Ajoutez la qualité de l'image : ").strip() or "50")
negative_prompt = str(input("(Optionel) Ajoutez un élément à ne pas avoir sur l'image (negative_prompt) : "))

# Générer et sauvegarder l'image
txt2ImgPost(prompt, guidance_scale, num_inference_steps, negative_prompt)