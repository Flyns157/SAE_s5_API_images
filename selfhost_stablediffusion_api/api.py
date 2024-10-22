from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import Blueprint, jsonify, request, send_file
from PIL import Image
import logging
import io

logger = logging.getLogger(__name__)

bp = Blueprint(name="gen_api", import_name=__name__, url_prefix="/api")

@bp.route('/protected', methods=['GET'])
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
@bp.route('/process-image', methods=['POST'])
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
