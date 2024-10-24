import torch
from diffusers import StableDiffusionPipeline
import random
from PIL import Image  
from utils import Utils
from peft import get_lora_model, LoraConfig, TaskType


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
        generator = torch.Generator(Utils.get_divice()).manual_seed(seed)   # type: ignore
        image = pipe(prompt, generator=generator, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]
        return image

    @staticmethod
    def txt2img_avatar(picture: Image, picture_type: "people" | "animal" | "background", people_description: str, animal_description: str, background_description: str, artist: str, pipe: StableDiffusionPipeline = Utils.load_SD_pipe(name="CompVis/stable-diffusion-v1-4")):
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
            case _:
                pass
        
        if picture == "painting":
            prompt += f" by {artist}"
        
        seed = random.randint(1, 100)
        generator = torch.Generator(Utils.get_divice()).manual_seed(seed) 
        image = pipe(prompt, generator=generator).images[0]
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
        Txt2Img.txt2ImgPost(prompt, guidance_scale,
                            num_inference_steps, negative_prompt)

    @staticmethod
    def apply_lora_finetuning(images: list[Image.Image], pipeline: StableDiffusionPipeline):
        # Ensure we are using the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 2: Set up LoRA configuration
        lora_config = LoraConfig(
            # Define the task type (Text-to-Image generation)
            task_type=TaskType.TEXT_TO_IMAGE,
            # Rank (size of the low-rank matrices)
            r=16,
            lora_alpha=32,                     # Scaling factor
            lora_dropout=0.05,                 # Dropout probability
            # Example of layers to apply LoRA
            target_modules=["transformer", "attention"],
            bias="none"                        # Fine-tune without modifying bias terms
        )

        # Step 3: Apply LoRA to the model inside the pipeline
        lora_model = get_lora_model(pipeline.model, lora_config)

        # Step 4: Fine-tune the model with the images
        fine_tuned_model = Txt2Img.fine_tune_model_with_images(
            lora_model, images, device)

        # Step 5: Update the pipeline with the fine-tuned model
        fine_tuned_pipeline = pipeline
        fine_tuned_pipeline.model = fine_tuned_model

        # Return the fine-tuned pipeline
        return fine_tuned_pipeline

    @staticmethod
    def fine_tune_model_with_images(lora_model, images, device):
        """
        Perform the fine-tuning process using the LoRA-adapted model and the user's images.
        """
        # Step 1: Transfer model to the appropriate device
        lora_model.to(device)

        # Step 2: Set up optimizer and training loop
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)

        # Example training loop (simplified)
        for epoch in range(5):  # For simplicity, we train for 5 epochs
            for img in images:
                # Step 3: Convert image to tensor and move to device
                img_tensor = Txt2Img.preprocess_image_to_tensor(img).to(device)

                # Forward pass (obtain model output for the input image)
                output = lora_model(img_tensor)

                # Calculate loss (use a loss function like MSE or other for fine-tuning)
                # Placeholder for loss computation
                loss = Txt2Img.compute_loss(output, img_tensor)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

        return lora_model  # Return the fine-tuned model

    @staticmethod
    def preprocess_image_to_tensor(image: Image.Image):
        """
        Convert an image into a tensor for training.
        """
        # Resize the image and convert it into a PyTorch tensor
        # Example: resizing all images to 512x512
        tensor = torch.tensor(image.resize((512, 512)))
        # Reorder dimensions for PyTorch (C, H, W)
        tensor = tensor.permute(2, 0, 1)
        # Normalize the pixel values to [0, 1]
        tensor = tensor.float() / 255.0
        return tensor

    def compute_loss(output, target):
        """
        Compute the loss between the model's output and the target image.
        """
        loss_fn = torch.nn.MSELoss()  # Mean Squared Error Loss for simplicity
        return loss_fn(output, target)
