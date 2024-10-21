from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from neo4j import GraphDatabase
import bcrypt

app = Flask(__name__)

# Configuration Neo4j
NEO4J_URI = "bolt://localhost:7687"  # URI de votre serveur Neo4j
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "yourpassword"

# Connexion à la base de données Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# JWT secret key
app.config['JWT_SECRET_KEY'] = 'votre_cle_secrete'  # Changez cette valeur pour plus de sécurité

# Initialiser le JWTManager avec l'application Flask
jwt = JWTManager(app)

# Fonction utilitaire pour hacher les mots de passe
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Fonction utilitaire pour vérifier un mot de passe haché
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Création d'un nouvel utilisateur dans Neo4j
def create_user(username, password):
    hashed_password = hash_password(password)
    query = "CREATE (u:User {username: $username, password: $password})"
    with driver.session() as session:
        session.run(query, username=username, password=hashed_password.decode('utf-8'))

# Recherche d'un utilisateur par nom dans Neo4j
def find_user_by_username(username):
    query = "MATCH (u:User {username: $username}) RETURN u.username AS username, u.password AS password"
    with driver.session() as session:
        result = session.run(query, username=username)
        return result.single()

# Route pour créer un nouvel utilisateur
@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    if not username or not password:
        return jsonify({"msg": "Nom d'utilisateur et mot de passe requis"}), 400

    # Vérifier si l'utilisateur existe déjà
    if find_user_by_username(username):
        return jsonify({"msg": "Utilisateur déjà existant"}), 400

    # Créer un nouvel utilisateur dans Neo4j
    create_user(username, password)
    return jsonify({"msg": f"Utilisateur {username} créé avec succès"}), 201

# Route pour se connecter et obtenir un token
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    # Récupérer l'utilisateur dans Neo4j
    user = find_user_by_username(username)

    if not user or not check_password(password, user["password"].encode('utf-8')):
        return jsonify({"msg": "Mauvais identifiants"}), 401

    # Créer un token JWT pour l'utilisateur
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

# Route protégée nécessitant un token JWT


if __name__ == '__main__':
    app.run(debug=True)
