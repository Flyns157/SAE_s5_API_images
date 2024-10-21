from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import Blueprint, jsonify
import logging

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__, url_prefix="/api")

@bp.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    # Récupérer l'identité de l'utilisateur à partir du token
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200
