from dotenv import load_dotenv
import secrets
import random
import string
import os

# Load environment variables from a .env file
load_dotenv()

def generate_password(size: int = 15) -> str:
    """
    Generate a random password of a given size.

    Parameters:
    size (int): The length of the password to be generated. Defaults to 15.

    Returns:
    str: A randomly generated password consisting of ASCII letters and digits.
    """
    CHARS = string.ascii_letters + string.digits
    return ''.join(random.choice(CHARS) for _ in range(size))

class Config:
    """
    Configuration class for the application settings.
    """
    SECRET_KEY = os.getenv('SECRET_KEY') if os.getenv('SECRET_KEY') and os.getenv('SECRET_KEY').lower() != 'auto' else secrets.token_urlsafe()
    SECURITY_PASSWORD_SALT = os.getenv('SECURITY_PASSWORD_SALT') if os.getenv('SECURITY_PASSWORD_SALT') and os.getenv('SECURITY_PASSWORD_SALT').lower() != 'auto' else secrets.token_hex(16)
    INDEPENDENT_REGISTER = str(os.getenv('INDEPENDENT_REGISTER') or 'True').lower() == 'true'
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY') if os.getenv('JWT_SECRET_KEY') and os.getenv('JWT_SECRET_KEY').lower() != 'auto' else secrets.token_hex(16)
    NEO4J_URI = os.getenv('NEO4J_URI') or 'bolt://localhost:7687'
    NEO4J_USER = os.getenv('NEO4J_USER') or 'neo4j'
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD') or 'neo4j'
    NEO4J_AUTH = os.getenv('NEO4J_USER')
    NO_AUTH = bool(os.getenv('NO_AUTH'))
