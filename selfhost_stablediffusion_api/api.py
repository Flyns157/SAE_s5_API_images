from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DiffusionPipeline
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename
from . import GenerationAPI
from PIL import Image
import logging
import base64
import io
import os

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
        processed_image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

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
            fav_hero=fav_hero,
            #pipe=GenerationAPI.get_pipeline(model_name="CompVis/stable-diffusion-v1-4")
        )

        img_io = io.BytesIO()
        image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#  ================================================================================================================================================

# from .generator import Inpainting

inpainting_bp = Blueprint(name="inpainting_api", import_name=__name__, url_prefix="/inpainting")

@inpainting_bp.route('/inpainting', methods=['POST'])
def inpainting():
    try:
        data = request.json
        strategy = int(data['strategy'])
        prompt = data['prompt']
        init_image = data['init_image']
        mask_image = data.get('mask_image', None)
        
        # Retrieve input images
        init_image_file = request.files['image']
        init_image = Image.open(init_image_file)
        mask_image_file = request.files.get('mask_image', None)
        mask_image = Image.open(init_image_file) if mask_image_file else None
        
        # Call the inpainting_choice function
        image = Image.new('RGB', (512, 512))# Inpainting.inpainting_choice(strategy=strategy,
        #                                         prompt=prompt,
        #                                         init_image=init_image,
        #                                         mask_image=mask_image,
        #                                         )
        
        # Sauvegarder l'image dans un buffer en mémoire
        img_io = io.BytesIO()
        image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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
                                        # pipe=GenerationAPI.get_pipeline(model_name="CompVis/stable-diffusion-v1-4", loader=StableDiffusionImg2ImgPipeline)
                                        )

        # Save the image in a memory buffer
        img_io = io.BytesIO()
        output_image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

#  ================================================================================================================================================

fine_inpainting_bp = Blueprint(name="fine-inpainting_api", import_name=__name__, url_prefix="/fine-inpainting")

from .generator import FineInpainting

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(base64_string, save_path):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    with open(save_path, 'wb') as f:
        f.write(image_data)
    return save_path

def image_to_base64(image):
    if isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    return None

@fine_inpainting_bp.route('', methods=['POST'])
def process_image():
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        required_fields = ['training_images', 'train_prompt_instance', 'class_prompt', 'input_image', 'user_prompt']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create temporary directory for this request
        request_dir = os.path.join(UPLOAD_FOLDER, secure_filename(str(os.urandom(8).hex())))
        os.makedirs(request_dir, exist_ok=True)

        try:
            # Save training images
            training_paths = []
            for i, img_data in enumerate(data['training_images']):
                training_path = os.path.join(request_dir, f'training_{i}.png')
                save_base64_image(img_data, training_path)
                training_paths.append(training_path)

            # Save input image
            input_path = os.path.join(request_dir, 'input.png')
            save_base64_image(data['input_image'], input_path)

            # Save mask if provided
            mask_path = None
            if 'user_mask' in data and data['user_mask']:
                mask_path = os.path.join(request_dir, 'mask.png')
                save_base64_image(data['user_mask'], mask_path)

            # Process the image
            result_image = FineInpainting().process_image(
                tab_train_image=training_paths,
                train_prompt_instance=data['train_prompt_instance'],
                class_prompt=data['class_prompt'],
                init_image=input_path,
                user_prompt=data['user_prompt'],
                user_mask=mask_path,
                strength=data.get('strength', 0.6)
            )
            
            if not result_image:
                return jsonify({'error': 'Failed to process image'}), 500
            
            # Save the image in a memory buffer
            img_io = io.BytesIO()
            result_image.save(img_io, format='PNG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')

        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(request_dir, ignore_errors=True)

    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500
