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

    Attributes:
    WTF_CSRF_ENABLED (bool): to activate CSRF protection.
    WTF_CSRF_SECRET_KEY (str): Secret key for CSRF protection.
    SQLALCHEMY_DATABASE_URI (str): URI for the database connection.
    SECRET_KEY (str): Secret key for session management.
    SECURITY_PASSWORD_SALT (str): Salt for password hashing.
    MAIL_DEBUG (bool): Flag for mail debugging.
    MAIL_SERVER (str): Mail server address.
    MAIL_PORT (str): Mail server port.
    MAIL_USE_TLS (bool): Flag for using TLS for mail.
    MAIL_USE_SSL (bool): Flag for using SSL for mail.
    MAIL_USERNAME (str): Username for the mail server.
    MAIL_PASSWORD (str): Password for the mail server.
    MAIL_DEFAULT_SENDER (tuple): Default sender name and email.
    DEFAULT_LANGUAGE (str): Default language for the application.
    BABEL_DEFAULT_LOCALE (str): Default locale for Babel.
    BABEL_DEFAULT_TIMEZONE (str): Default timezone for Babel.
    ADMIN_PASSWORD (str): Admin password for the application.
    MAIL_SERVICES (bool): Flag for enabling mail services.
    V2F (int): Custom configuration variable.
    INDEPENDENT_REGISTER (bool): Flag for independent registration setting.
    """
    WTF_CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = generate_password() if not os.getenv('WTF_CSRF_SECRET_KEY') or os.getenv('WTF_CSRF_SECRET_KEY').upper() == 'AUTO' else os.getenv('WTF_CSRF_SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI') or 'sqlite:///db.sqlite'
    SECRET_KEY = os.getenv('SECRET_KEY') if os.getenv('SECRET_KEY') and os.getenv('SECRET_KEY').lower() != 'auto' else secrets.token_urlsafe()
    SECURITY_PASSWORD_SALT = os.getenv('SECURITY_PASSWORD_SALT') if os.getenv('SECURITY_PASSWORD_SALT') and os.getenv('SECURITY_PASSWORD_SALT').lower() != 'auto' else secrets.token_hex(16)
    MAIL_DEBUG = os.getenv('MAIL_DEBUG').lower() == 'true'
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_PORT = os.getenv('MAIL_PORT')
    MAIL_USE_TLS = os.getenv('MAIL_USE').upper() == "TLS"
    MAIL_USE_SSL = os.getenv('MAIL_USE').upper() == "SSL"
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = ('Autadash', os.getenv('MAIL_DEFAULT_SENDER'))
    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE') or 'en'
    BABEL_DEFAULT_LOCALE = str(os.getenv('INDEPENDENT_REGISTER')) or 'translations'
    BABEL_DEFAULT_TIMEZONE = os.getenv('DEFAULT_TIMEZONE') or 'UTC'
    ADMIN_PASSWORD = generate_password() if not os.getenv('ADMIN_PASSWORD') or os.getenv('ADMIN_PASSWORD').upper() == 'AUTO' else os.getenv('ADMIN_PASSWORD')
    MAIL_SERVICES = str(os.getenv('MAIL_SERVICES') or 'False').lower() == 'true'
    V2F = int(os.getenv('V2F') or '0')
    INDEPENDENT_REGISTER = str(os.getenv('INDEPENDENT_REGISTER') or 'True').lower() == 'true'
