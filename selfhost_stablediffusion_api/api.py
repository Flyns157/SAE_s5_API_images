from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DiffusionPipeline
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import Blueprint, jsonify, request, send_file
from . import GenerationAPI
from PIL import Image
import logging
import io

logger = logging.getLogger(__name__)

example_bp = Blueprint(name="gen_api", import_name=__name__, url_prefix="/api")

@example_bp.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    # Récupérer l'identité de l'utilisateur à partir du token
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

# Exemple de fonction de traitement d'image
def process_image(image):
    '''
    Convertir l'image en noir et blanc
    '''
    return image.convert('L')

# Route API pour envoyer l'image et recevoir l'image modifiée (en noir et blanc ici)
@example_bp.route('/process-image', methods=['POST'])
def process_image_route():
    if 'image' not in request.files:
        return "No image file provided", 400

    file = request.files['image']
    
    try:
        image = Image.open(file.stream)
        
        processed_image = process_image(image)
        
        # mise en cache de l'image (buffer)
        img_io = io.BytesIO()
        processed_image.save(img_io, format='JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return f"Error processing image: {str(e)}", 500

#  ================================================================================================================================================

txt2img_bp = Blueprint(name="txt-img_api", import_name=__name__, url_prefix="/txt2img")

from .generator import Txt2Img

@txt2img_bp.route('/post', methods=['POST'])
def generate_txt2img():
    try:
        data = request.json
        prompt = data['prompt']
        guidance_scale = float(data.get('guidance_scale', 7.5))
        num_inference_steps = int(data.get('num_inference_steps', 50))
        negative_prompt = data.get('negative_prompt', "")

        # Utilisation de la méthode Txt2Img.txt2ImgPost
        image = Txt2Img.txt2img_post(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            # pipe=GenerationAPI.get_pipeline(model_name="CompVis/stable-diffusion-v1-4")
        )

        # Sauvegarder l'image dans un buffer en mémoire
        img_io = io.BytesIO()
        image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@txt2img_bp.route('/avatar', methods=['POST'])
def generate_avatar():
    try:
        data = request.json
        picture = data['picture']
        picture_type = data['picture_type']
        people_description = data.get('people_description', ["", "", "", "", "", "", "", ""])
        animal_description = data.get('animal_description', ["", ""])
        background_description = data.get('background_description', "")
        artist = data.get('artist', "")

        # Utilisation de la méthode Txt2Img.txt2ImgAvatar
        image = Txt2Img.txt2img_avatar(
            picture=picture,
            picture_type=picture_type,
            people_description=people_description,
            animal_description=animal_description,
            background_description=background_description,
            artist=artist,
            #pipe=GenerationAPI.get_pipeline(model_name="CompVis/stable-diffusion-v1-4")
        )

        # Sauvegarder l'image dans un buffer en mémoire
        img_io = io.BytesIO()
        image.save(img_io, format='JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#  ================================================================================================================================================

img2img_bp = Blueprint(name="img-img_api", import_name=__name__, url_prefix="/img2img")

from .generator import Img2Img

@img2img_bp.route('', methods=['POST'])
def generate_img2img():
    try:
        # Retrieving data from the query
        prompt = request.form['prompt']
        strength = float(request.form.get('strength', 0.75))
        num_inference_steps = int(request.form.get('num_inference_steps', 50))

        # Récupérer l'image d'entrée
        init_image_file = request.files['image']
        init_image = Image.open(init_image_file)

        # Load the img2img pipeline using the img2img method
        output_image = Img2Img.img2img(prompt=prompt,
                                        init_image=init_image,
                                        strength=strength,
                                        num_inference_steps=num_inference_steps,
                                        pipe=GenerationAPI.get_pipeline(model_name="CompVis/stable-diffusion-v1-4", loader=StableDiffusionImg2ImgPipeline))

        # Save the image in a memory buffer
        img_io = io.BytesIO()
        output_image.save(img_io, format='JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#  ================================================================================================================================================
