"""
Add authentication routes to the application.
"""
from flask_jwt_extended import create_access_token
from flask import Blueprint, jsonify, request
from .utils import Database
from .config import Config
from . import db

config = Config()

bp = Blueprint("api", __name__, url_prefix="/api")

if config.INDEPENDENT_REGISTER:
    @bp.route('/register', methods=['POST'])
    # @bp.limiter.limit("5 per minute")
    def register():
        username = request.json.get('username', None)
        password = request.json.get('password', None)
        if not username or not password:
            return jsonify({"msg": "Nom d'utilisateur et mot de passe requis"}), 400
        if db.find_user_by_username(username):
            return jsonify({"msg": "Utilisateur déjà existant"}), 400
        db.create_user(username, password)
        return jsonify({"msg": f"Utilisateur {username} créé avec succès"}), 201

@bp.route('/login', methods=['POST'])
# @bp.limiter.limit("10 per minute")
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    user = db.find_user_by_username(username)
    if not user or not Database.check_password(password, user["password"].encode('utf-8')):
        return jsonify({"msg": "Mauvais identifiants"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)
