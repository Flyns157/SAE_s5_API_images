from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import bcrypt

app = Flask(__name__)

# Configuration de la base de données PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://flask_user:yourpassword@localhost/flask_auth'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'votre_cle_secrete'  # Clé secrète pour JWT

# Initialiser les extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)

# Modèle pour la table des utilisateurs
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Fonction utilitaire pour hacher les mots de passe
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Fonction utilitaire pour vérifier un mot de passe haché
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Route pour créer un nouvel utilisateur (pour initialisation)
@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    if not username or not password:
        return jsonify({"msg": "Nom d'utilisateur et mot de passe requis"}), 400

    # Vérifier si l'utilisateur existe déjà
    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Utilisateur déjà existant"}), 400

    # Hacher le mot de passe et créer un nouvel utilisateur
    hashed_password = hash_password(password)
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"msg": f"Utilisateur {username} créé avec succès"}), 201

# Route pour se connecter et obtenir un token
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    # Récupérer l'utilisateur dans la base de données
    user = User.query.filter_by(username=username).first()

    if not user or not check_password(password, user.password):
        return jsonify({"msg": "Mauvais identifiants"}), 401

    # Créer un token JWT pour l'utilisateur
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

# Route protégée nécessitant un token JWT
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    # Récupérer l'identité de l'utilisateur à partir du token
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

if __name__ == '__main__':
    # Créer toutes les tables dans la base de données
    with app.app_context():
        db.create_all()  # Création des tables si elles n'existent pas
    app.run(debug=True)
