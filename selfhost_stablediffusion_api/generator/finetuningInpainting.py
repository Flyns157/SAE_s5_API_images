import subprocess
import os
from PIL import Image
import torch
import random
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration, CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline
import cv2
import io
import base64

class ImageProcessor:
    def __init__(self, device=None):
        self.device = device or self._get_device()
        self.initialize_models()
        
    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize_models(self):
        # Initialize CLIP model
        self.clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        
        # Initialize Kosmos model
        self.kosmos_model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        
    def _create_image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows * cols
        
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
        
    def generate_mask_image(self, init_image, mask_prompt, out_fpath=None, debug=False):
        if isinstance(mask_prompt, str):
            mask_prompt = [mask_prompt, mask_prompt]
        if isinstance(mask_prompt, list) and len(mask_prompt) == 1:
            mask_prompt = mask_prompt * 2
            
        inputs = self.clip_processor(
            text=mask_prompt, 
            images=[init_image] * len(mask_prompt), 
            padding="max_length", 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            
        preds = outputs.logits.unsqueeze(1)
        
        if debug:
            self._visualize_predictions(init_image, preds, mask_prompt)
            
        temp_fpath = "temp.png"
        plt.imsave(temp_fpath, torch.sigmoid(preds[1][0]))
        img2 = cv2.imread(temp_fpath)
        gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        _, bw_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
        
        mask_image = Image.fromarray(bw_image)
        if out_fpath:
            mask_image.save(out_fpath)
        return mask_image
        
    def _visualize_predictions(self, init_image, preds, mask_prompt):
        _, ax = plt.subplots(1, 5, figsize=(15, len(mask_prompt)+1))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(init_image)
        [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(mask_prompt))]
        [ax[i+1].text(0, -15, mask_prompt[i]) for i in range(len(mask_prompt))]
        
    def _load_image(self, image_data):
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        return Image.open(image_data)
        
    def _setup_training_environment(self, tab_train_image):
        subprocess.run(['gdown', 'https://drive.google.com/uc?id=1CC0YdBBPjqy1dyW6kNfH5qwmLJnlXGhR'], check=True)
        os.makedirs("/content/dreambooth_inpainting_out", exist_ok=True)
        save_dir = "/content/dreambooth_inpainting_in"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, image_data in enumerate(tab_train_image):
            img = self._load_image(image_data)
            img_save_path = os.path.join(save_dir, f"{i}.jpeg")
            img.save(img_save_path)
            
    def _train_model(self, train_prompt_instance, class_prompt):
        command = f"""accelerate launch train_dreambooth_inpaint.py 
            --pretrained_model_name_or_path=$MODEL_NAME 
            --instance_data_dir=$INSTANCE_DIR 
            --class_data_dir=$CLASS_DIR 
            --output_dir=$OUTPUT_DIR 
            --with_prior_preservation --prior_loss_weight=1.0 
            --instance_prompt={train_prompt_instance} 
            --class_prompt={class_prompt} 
            --resolution=512 
            --train_batch_size=1 
            --gradient_accumulation_steps=1 
            --use_8bit_adam 
            --learning_rate=5e-6 
            --lr_scheduler="constant" 
            --lr_warmup_steps=0 
            --num_class_images=200 
            --max_train_steps=400 
            --train_text_encoder 
            --checkpointing_steps=4000"""
            
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout, stderr
        
    def generate_entities(self, init_image, prompt="<grounding> An image of"):
        inputs = self.kosmos_processor(
            text=prompt, 
            images=init_image, 
            return_tensors="pt"
        )
        
        generated_ids = self.kosmos_model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=64,
        )
        
        generated_text = self.kosmos_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption, entities = self.kosmos_processor.post_process_generation(generated_text)
        return entities
        
    def _localized_blend_images(self, base_image, generated_images, masks):
        base_image = base_image.convert("RGBA")
        final_image = base_image.copy()
        
        for img, mask in zip(generated_images, masks):
            img_rgba = img.convert("RGBA")
            img_rgba = img_rgba.resize(base_image.size)
            mask = mask.resize(base_image.size)
            mask_alpha = mask.convert("L")
            final_image = Image.composite(img_rgba, final_image, mask_alpha)
            
        return final_image.convert("RGB")
        
    def process_image(self, tab_train_image, train_prompt_instance, class_prompt, 
                     init_image, user_prompt, user_mask=None, strength=0.6):
        # Load and process initial image
        init_image = self._load_image(init_image)
        
        # Setup training environment and train model
        self._setup_training_environment(tab_train_image)
        stdout, stderr = self._train_model(train_prompt_instance, class_prompt)
        
        # Initialize pipeline
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "/content/dreambooth_inpainting_out",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Process masks and generate images
        if user_mask is None:
            # Generate masks from image entities
            entities = self.generate_entities(init_image)
            mask_prompt = [item[0] for item in entities]
            masks = [self.generate_mask_image(init_image, word) for word in mask_prompt]
            print(f"Elements found in image: {mask_prompt}")
            
            generated_images = []
            for mask_word in mask_prompt:
                print(f"Prompt for {mask_word}: {user_prompt}")
                generator = torch.Generator(self.device).manual_seed(random.randint(1, 9999999))
                
                result_image = pipeline(
                    prompt=user_prompt,
                    image=init_image,
                    mask_image=self.generate_mask_image(init_image, mask_word),
                    generator=generator,
                    strength=strength
                ).images[0]
                
                generated_images.append(result_image)
        else:
            # Use provided mask
            mask_image = self._load_image(user_mask)
            generator = torch.Generator(self.device).manual_seed(random.randint(1, 9999999))
            
            result_image = pipeline(
                prompt=user_prompt,
                image=init_image,
                mask_image=mask_image,
                generator=generator,
                strength=strength
            ).images[0]
            
            generated_images = [result_image]
            masks = [mask_image]
        
        # Blend images and visualize results
        final_image = self._localized_blend_images(init_image, generated_images, masks)
        
        self._visualize_results(init_image, generated_images, final_image)
        
        return final_image
        
    def _visualize_results(self, init_image, generated_images, final_image):
        fig, axs = plt.subplots(1, len(generated_images) + 2, figsize=(20, 5))
        axs[0].imshow(init_image)
        axs[0].set_title('Input Image')
        
        for i, img in enumerate(generated_images):
            axs[i+1].imshow(img)
            axs[i+1].set_title(f'Result {i+1}')
            
        axs[-1].imshow(final_image)
        axs[-1].set_title('Final Localized Blended Image')
        plt.show()

# Example usage:
# processor = ImageProcessor()
# final_image = processor.process_image(
#     tab_train_image=["0.jpeg","1.jpeg","2.jpeg","3.jpeg","4.jpeg"],
#     train_prompt_instance="a photo of sks dog",
#     class_prompt="a photo of dog",
#     init_image="haci.jpg",
#     user_prompt="a beautiful dog"
# )
