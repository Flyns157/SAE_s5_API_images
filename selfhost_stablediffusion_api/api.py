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






txt2img_bp = Blueprint(name="txt-img_api", import_name=__name__, url_prefix="/txt2img")

from generator import Txt2Img

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
            pipe=GenerationAPI.get_pipeline(model_name="CompVis/stable-diffusion-v1-4")
        )

        # Sauvegarder l'image dans un buffer en mémoire
        img_io = io.BytesIO()
        image.save(img_io, format='JPEG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@txt2img_bp.route('/avatar', methods=['POST'])
def generate_avatar():
    try:
        # Extract data from request JSON
        data = request.json
        
        # Extract all parameters, using defaults for missing ones
        image_type = data.get('image_type', 'picture')
        style = data.get('style', None)
        subject = data.get('subject', None)
        gender = data.get('gender', None)
        hair_color = data.get('hair_color', None)
        hair_length = data.get('hair_length', None)
        haircut = data.get('haircut', None)
        nationality = data.get('nationality', None)
        eye_color = data.get('eye_color', None)
        animal = data.get('animal', None)
        body_color = data.get('body_color', None)
        height = data.get('height', None)
        environment = data.get('environment', None)
        fav_color = data.get('fav_color', None)
        fav_sport = data.get('fav_sport', None)
        fav_animal = data.get('fav_animal', None)
        fav_song = data.get('fav_song', None)
        fav_dish = data.get('fav_dish', None)
        fav_job = data.get('fav_job', None)
        fav_hero = data.get('fav_hero', None)
        
        # Generate avatar using the txt2img_avatar function
        image = Txt2Img.txt2img_avatar(
            image_type=image_type,
            style=style,
            subject=subject,
            gender=gender,
            hair_color=hair_color,
            hair_length=hair_length,
            haircut=haircut,
            nationality=nationality,
            eye_color=eye_color,
            animal=animal,
            body_color=body_color,
            height=height,
            environment=environment,
            fav_color=fav_color,
            fav_sport=fav_sport,
            fav_animal=fav_animal,
            fav_song=fav_song,
            fav_dish=fav_dish,
            fav_job=fav_job,
            fav_hero=fav_hero
        )

        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
