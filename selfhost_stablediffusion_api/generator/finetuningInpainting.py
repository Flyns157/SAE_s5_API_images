import subprocess
import os
from PIL import Image
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import PIL
from io import BytesIO
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration, CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline
import cv2


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# We'll use a couple of demo images later in the notebook
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


processor2 = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model2 = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# 'mask_prompt' must have at least 2 keywords
def generate_mask_image(init_image, mask_prompt, out_fpath, debug=False):
    temp_fpath = f"temp.png"
    if isinstance(mask_prompt, str):
      mask_prompt = [mask_prompt, mask_prompt]
    if isinstance(mask_prompt, list) and len(mask_prompt) == 1:
      mask_prompt = mask_prompt * 2
    inputs = processor2(text=mask_prompt, images=[init_image] * len(mask_prompt), padding="max_length", return_tensors="pt")

    # predict
    with torch.no_grad():
      outputs = model2(**inputs)

    preds = outputs.logits.unsqueeze(1)

    # visualize prediction
    if debug:
        _, ax = plt.subplots(1, 5, figsize=(15, len(mask_prompt)+1))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(init_image)
        print(torch.sigmoid(preds[0][0]).shape)
        [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(mask_prompt))];
        [ax[i+1].text(0, -15, mask_prompt[i]) for i in range(len(mask_prompt))];

    plt.imsave(temp_fpath,torch.sigmoid(preds[1][0]))
    img2 = cv2.imread(temp_fpath)
    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # fix color format
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

    mask_image = PIL.Image.fromarray(bw_image)
    mask_image.save(out_fpath)
    return mask_image


model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

def process_image(tab_train_image, train_prompt_instance, class_prompt, init_image, userPrompt, userMask=None, strength=0.6):
    if ',' in init_image:
        init_image = init_image.split(',')[1]
        init_image = base64.b64decode(init_image)
        init_image = Image.open(io.BytesIO(init_image))
    else:
        # Si l'image n'est pas en base64, chargez-la directement
        init_image = Image.open(init_image)


    mask_image = None
    if userMask:
        if ',' in userMask:
            userMask = userMask.split(',')[1]
            userMask = base64.b64decode(userMask)
            mask_image = Image.open(io.BytesIO(userMask))
        else:
            mask_image = Image.open(userMask)

    subprocess.run(['gdown', 'https://drive.google.com/uc?id=1CC0YdBBPjqy1dyW6kNfH5qwmLJnlXGhR'], check=True)
    os.makedirs("/content/dreambooth_inpainting_out", exist_ok=True)
    save_dir = "/content/dreambooth_inpainting_in"
    os.makedirs(save_dir, exist_ok=True)

    for i, image_data in enumerate(tab_train_image):
        if ',' in image_data:
            # Convertir l'image base64 en Image PIL
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
        else:
            # Si l'image n'est pas en base64, chargez-la directement
            img = Image.open(image_data)

        # Sauvegarder l'image dans le dossier
        img_save_path = os.path.join(save_dir, f"{i}.jpeg")
        img.save(img_save_path)

    !accelerate launch train_dreambooth_inpaint.py \
        --pretrained_model_name_or_path=$MODEL_NAME  \
        --instance_data_dir=$INSTANCE_DIR \
        --class_data_dir=$CLASS_DIR \
        --output_dir=$OUTPUT_DIR \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --instance_prompt=train_prompt_instance \
        --class_prompt=class_prompt \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --use_8bit_adam \
        --learning_rate=5e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=400 \
        --train_text_encoder \
        --checkpointing_steps=4000

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "/content/dreambooth_inpainting_out",
            torch_dtype=torch.float16
        ).to("cuda")

    def generate_entities(init_image, prompt="<grounding> An image of"):
        model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

        inputs = processor(text=prompt, images=init_image, return_tensors="pt")

        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=64,
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        caption, entities = processor.post_process_generation(generated_text)

        return entities

    def generate_individual_mask(init_image, mask_word):
        out_fpath = f"mask_{mask_word}.png"
        mask_image = generate_mask_image(init_image, [mask_word], out_fpath, debug=False)
        return mask_image

    if mask_image is None:
        texte = [item[0] for item in generate_entities(init_image)]
        mask_prompt = texte  # It should be a list of words
        mask = [generate_individual_mask(init_image, word) for word in mask_prompt]
        print(f"Element trouver dans l'image : {mask_prompt}")

        generated_images = []
        for mask_word in mask_prompt:
            print(f"Prompt pour {mask_word} : {userPrompt}")

            seed = random.randint(1, 9999999)
            generator = torch.Generator("cuda").manual_seed(seed)

            result_image = pipeline(
                prompt=userPrompt,
                image=init_image,
                mask_image=generate_individual_mask(init_image, mask_word),
                generator=generator,
                strength=strength
            ).images[0]

            generated_images.append(result_image)
    else:
        generated_images = []
        print(f"Prompt pour : {userPrompt}")
        mask = mask_image

        seed = random.randint(1, 9999999)
        generator = torch.Generator("cuda").manual_seed(seed)

        result_image = pipeline(
            prompt=userPrompt,
            image=init_image,
            mask_image=mask,
            generator=generator,
            strength=strength
        ).images[0]

        generated_images.append(result_image)

    def localized_blend_images(base_image, generated_images, masks):
        base_image = base_image.convert("RGBA")
        final_image = base_image.copy()

        for img, mask in zip(generated_images, masks):
            img_rgba = img.convert("RGBA")

            img_rgba = img_rgba.resize(base_image.size)
            mask = mask.resize(base_image.size)

            mask_alpha = mask.convert("L")

            final_image = Image.composite(img_rgba, final_image, mask_alpha)

        return final_image.convert("RGB")

    final_image = localized_blend_images(init_image, generated_images, mask)

    fig, axs = plt.subplots(1, len(generated_images) + 2, figsize=(20, 5))
    axs[0].imshow(init_image); axs[0].set_title('Input Image')
    for i, img in enumerate(generated_images):
        axs[i+1].imshow(img); axs[i+1].set_title(f'Result {i+1}')
    axs[-1].imshow(final_image); axs[-1].set_title('Final Localized Blended Image')
    plt.show()

    return final_image

#test
#im = process_image(["0.jpeg","1.jpeg","2.jpeg","3.jpeg","4.jpeg"],"a photo of sks dog","a photo of dog","haci.jpg","a beautiful dog")