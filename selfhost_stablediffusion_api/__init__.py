from flask import Flask, render_template, request, flash, url_for, redirect, render_template_string, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter.util import get_remote_address
from itsdangerous import URLSafeTimedSerializer
from .forms import RegistrationForm, LoginForm
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_wtf import CSRFProtect
from flask_limiter import Limiter
from flask_babel import Babel, _
from .config import Config
import logging
import random
import string
import json
import re

class Server(Flask):
    """
    Server class extends Flask to provide additional functionality such as
    configuration loading, user authentication, database setup, and email handling.

    Attributes:
    languages (list): List of supported languages.
    babel (Babel): Instance of Babel for i18n support.
    login_manager (LoginManager): Instance of LoginManager for user session management.
    mail (Mail): Instance of Mail for email handling.
    db (SQLAlchemy): Instance of SQLAlchemy for ORM and database interaction.
    logger (logging.Logger): Logger for logging application events.
    User (class): User model class.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the Server instance.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.config.from_object(Config)
        self.languages = kwargs.get('languages', ['en'])
        self.login_manager = LoginManager()
        self.limiter = Limiter(get_remote_address, default_limits=["200 per day", "50 per hour"])
        self.babel = Babel()
        self.csrf = CSRFProtect(self)
        self.mail = Mail()
        self.db = SQLAlchemy()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler('autadash.log'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)
        
        self.User = self.create_user_model()
        
        @self.login_manager.user_loader
        def load_user(user_id): return self.User.query.get(int(user_id))
        
        self.add_auth_routes()

    def run(self, host: str = None, port: int = None, debug: bool = None, load_dotenv: bool = True, **options) -> None:
        """
        Run the Flask application.

        Parameters:
        host (str): Hostname to listen on.
        port (int): Port to listen on.
        debug (bool): Enable or disable debug mode.
        load_dotenv (bool): Load environment variables from .env file.
        **options: Additional options to pass to the Flask run method.
        """
        # Initialize flask_mail
        self.mail.init_app(self)

        # Initialize flask_login
        self.login_manager.init_app(self)

        # Initialize flask_babel
        self.babel.init_app(self, locale_selector=lambda : request.accept_languages.best_match(self.languages))
        
        # Initialize flask_limiter
        self.limiter.init_app(self)

        # Initialize SQLAlchemy
        self.db.init_app(self)
        with self.app_context(): self.db.create_all()
        # TODO : Handle changes in parameters and modify the database if necessary
        return super().run(host, port, debug, load_dotenv, **options)

    def create_user_model(self):
        """
        Create the User model.

        Returns:
        User (class): The User model class.
        """
        global User
        class User(UserMixin, self.db.Model):
            """
            Data model for users.

            Attributes:
            id (int): User ID.
            username (str): Username.
            email (str): Email address.
            password (str): Password hash.
            known_devices (str): JSON string of known devices.
            email_confirmed (bool): Email confirmation status (V2F > 0).
            verification_code (str): Verification code (V2F > 0).
            verification_code_expiry (datetime): Expiry time of the verification code (V2F > 0).
            v2f (bool): 2-factor authentication status (V2F > 0).
            """

            id = self.db.Column(self.db.Integer, primary_key=True)
            username = self.db.Column(self.db.String(100), unique=True)
            email = self.db.Column(self.db.String(100), unique=True)
            password = self.db.Column(self.db.String(100))
            known_devices = self.db.Column(self.db.String, default='[]')  # Storing known devices as JSON string
            if self.config['V2F'] > 0:
                email_confirmed = self.db.Column(self.db.Boolean, default=False)
                verification_code = self.db.Column(self.db.String(6))
                verification_code_expiry = self.db.Column(self.db.DateTime)  # Column for verification code expiry
                v2f = self.db.Column(self.db.Boolean, default=False)

            def set_password(self, password):
                """
                Set the password for the user.

                Parameters:
                password (str): The password to be hashed and set.
                """
                self.password = generate_password_hash(password)

            def check_password(self, password):
                """
                Check the password against the stored hash.

                Parameters:
                password (str): The password to check.

                Returns:
                bool: True if the password matches, False otherwise.
                """
                return check_password_hash(self.password, password)

            def add_known_device(self, device_info):
                """
                Add a known device to the user's known devices list.

                Parameters:
                device_info (str): Information about the device.
                """
                devices = json.loads(self.known_devices)
                devices.append(device_info)
                self.known_devices = json.dumps(devices)

            def is_device_known(self, device_info):
                """
                Check if a device is known.

                Parameters:
                device_info (str): Information about the device.

                Returns:
                bool: True if the device is known, False otherwise.
                """
                devices = json.loads(self.known_devices)
                return device_info in devices
            
            def verification_code_alive(self) -> bool:
                """
                Check if the verification code is still valid.

                Returns:
                bool: True if the verification code is still valid, False otherwise.
                """
                if self.verification_code_expiry is None:
                    return False
                return self.verification_code_expiry > datetime.now()
        
        return User

    def add_auth_routes(self):
        """
        Add authentication routes to the application.
        """
        if self.config['INDEPENDENT_REGISTER']:
            @self.route('/register', methods=['GET', 'POST'])
            @self.limiter.limit("5 per minute")
            def register():
                """
                Handle user registration.
                
                Returns:
                Response: Rendered registration template or redirect response.
                """
                form = RegistrationForm()
                if form.validate_on_submit():
                    username = form.username.data
                    email = form.email.data
                    password = form.password.data

                    self.logger.info(f'User register attempt: username={username}, email={email}')

                    # Verify email format
                    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        flash(_('Invalid email format'))
                        self.logger.warning(f'Invalid email format: {email}')
                        return redirect(url_for('register'))

                    # Verify if the email is disposable
                    if email.split('@')[1] in ['jetable.com', 'yopmail.com']:
                        flash(_('Please use a non-disposable email address'))
                        self.logger.warning(f'Disposable email used: {email} by {username}')
                        return redirect(url_for('register'))

                    # Check if the email already exists
                    user = User.query.filter_by(email=email).first()
                    if user:
                        flash(_('Email is already in use'))
                        self.logger.warning(f'Email already in use: {email} ({username} try but {user} exist with tis email)')
                        return redirect(url_for('register'))

                    # Check if password is valid
                    if len(password) < 12 or not re.search("^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$", password):
                        flash('Le mot de passe doit contenir au moins 12 caractères, dont des chiffres, des lettres majuscules et minuscules.')
                        self.logger.warning(f'Password not enought stonger: username={username}, email={email}')
                        return redirect(url_for('register'))

                    # Check if username already exists
                    user = User.query.filter_by(username=username).first()
                    if user:
                        flash(_('Username already exists'))
                        self.logger.warning(f'Username already exists: {username} {"(with " + email + ")" if email != user.email else ""}')
                        return redirect(url_for('register'))

                    new_user = User(username=username, email=email)
                    new_user.set_password(password)
                    self.db.session.add(new_user)
                    self.db.session.commit()

                    self.logger.info(f'New user created: {username}, {email}')

                    if self.config['V2F'] > 0:
                        # Send confirmation email
                        token = self.generate_confirmation_token(new_user.email)
                        confirm_url = url_for('confirm_email', token=token, _external=True)
                        html = render_template('email/activate.html', confirm_url=confirm_url, SUPPORTED_LANGUAGES=self.languages)
                        self.send_email(new_user.email, 'Please confirm your email', html)

                        self.logger.info(f'Confirmation email sent to: {email}')

                        if self.config['V2F'] == 2:
                            return redirect(url_for('login'))

                        user = User.query.filter(User.email == email).first()
                        login_user(user)
                        self.logger.info(f'User logged in: {user.username}')
                        return redirect(url_for('/'))

                    login_user(user)
                    self.logger.info(f'User logged in: {user.username}')
                    return redirect(url_for('/'))
                return render_template('auth/register.html', form=form, SUPPORTED_LANGUAGES=self.languages)

        @self.route('/login', methods=['GET', 'POST'])
        @self.limiter.limit("10 per minute")
        def login():
            form = LoginForm()
            if form.validate_on_submit():
                username_or_email = form.username_or_email.data
                password = form.password.data

                self.logger.info(f'Login attempt: username_or_email={username_or_email}')

                user = User.query.filter((User.email == username_or_email) | (User.username == username_or_email)).first()
                if not user or not user.check_password(password):
                    self.logger.warning(f'Login failed for: {username_or_email}')
                    flash(_('Please check your login details and try again.'))
                    return redirect(url_for('login'))

                if self.config['V2F'] == 2:
                    if not user.email_confirmed:
                        self.logger.warning(f'Email not confirmed for: {username_or_email}')
                        flash(_('Please confirm your email address before logging in.'))
                        return redirect(url_for('login'))
                    self.send_verification_email(user)
                    self.logger.info(f'Verification code sent to: {user.email}')

                    return redirect(url_for('verify', user_id=user.id))

                login_user(user)
                self.logger.info(f'User logged in: {user.username}')
                return redirect(url_for('/'))

            return render_template('auth/login.html', form=form, SUPPORTED_LANGUAGES=self.languages)

        if self.config['V2F'] > 0:
            @self.route('/verify/<int:user_id>', methods=['GET', 'POST'])
            def verify(user_id):
                """
                Handle 2-factor verification.

                Parameters:
                user_id (int): The ID of the user to verify.

                Returns:
                Response: Rendered verification template or redirect response.
                """
                user = User.query.get(user_id)
                if request.method == 'POST':
                    verification_code = request.form.get('verification_code')
                
                    if user.verification_code != verification_code:
                        self.logger.warning(f'Invalid verification code for: {user.email}')
                        flash(_('Code de vérification incorrect'))
                        return redirect(url_for('verify', user_id=user_id))

                    if not user.verification_code_alive():
                        self.logger.warning(f'Expired verification code for: {user.email}')
                        flash(_('Code de vérification expiré'))
                        return redirect(url_for('login'))

                    # Clear the verification code and expiry after successful verification
                    user.verification_code = None
                    user.verification_code_expiry = None
                    self.db.session.commit()

                    login_user(user)
                    self.logger.info(f'User logged in: {user.username}')
                    return redirect(url_for('/'))

                return render_template('auth/verify.html', user_id=user_id, SUPPORTED_LANGUAGES=self.languages)

            @self.route('/reset_password_request', methods=['GET', 'POST'])
            def reset_password_request():
                """
                Handle password reset request.

                Returns:
                Response: Rendered password reset request template or redirect response.
                """
                if request.method == 'POST':
                    email = request.form.get('email')
                    user = User.query.filter_by(email=email).first()
                    if user:
                        token = self.generate_confirmation_token(user.email)
                        reset_url = url_for('reset_password', token=token, _external=True)
                        html = render_template('email/reset_password.html', reset_url=reset_url)
                        self.send_email(user.email, 'Réinitialisation du mot de passe', html)
                        flash(_('Un email avec les instructions pour réinitialiser votre mot de passe a été envoyé.'), 'info')
                    else:
                        flash(_('Cet email n\'est pas enregistré.'), 'warning')
                    return redirect(url_for('login'))
                return render_template('auth/reset_password_request.html', SUPPORTED_LANGUAGES=self.languages)

            @self.route('/reset_password/<token>', methods=['GET', 'POST'])
            def reset_password(token):
                """
                Handle password reset.

                Parameters:
                token (str): The token for password reset.

                Returns:
                Response: Rendered password reset template or redirect response.
                """
                try:
                    email = self.confirm_token(token)
                except:
                    flash(_('Le lien de réinitialisation est invalide ou a expiré.'), 'danger')
                    return redirect(url_for('login'))
                
                if request.method == 'POST':
                    password = request.form.get('password')
                    user = User.query.filter_by(email=email).first()
                    if user:
                        user.set_password(password)
                        self.db.session.commit()
                        flash(_('Votre mot de passe a été mis à jour.'), 'success')
                        return redirect(url_for('login'))
                    else:
                        flash(_('Utilisateur non trouvé.'), 'danger')
                        return redirect(url_for('reset_password_request'))
                
                return render_template('auth/reset_password.html', token=token, SUPPORTED_LANGUAGES=self.languages)

            @self.route('/confirm/<token>')
            def confirm_email(token):
                """
                Handle email confirmation.

                Parameters:
                token (str): The token for email confirmation.

                Returns:
                Response: Redirect response after email confirmation.
                """
                try:
                    email = self.confirm_token(token)
                except:
                    self.logger.error(f'Invalid or expired confirmation token used: {token}')
                    flash(_('Le lien de confirmation est invalide ou a expiré.'), 'danger')
                    return redirect(url_for('login'))

                user = User.query.filter_by(email=email).first_or_404()
                if user.email_confirmed:
                    self.logger.info(f'Email already confirmed: {email}')
                    flash(_('Compte déjà confirmé. Veuillez vous connecter.'), 'success')
                else:
                    user.email_confirmed = True
                    self.db.session.add(user)
                    self.db.session.commit()
                    self.logger.info(f'Email confirmed: {email}')
                    flash(_('Vous avez confirmé votre compte. Merci!'), 'success')
                return redirect(url_for('login'))

            @self.route('/send-confirmation-email', methods=['POST'])
            @login_required
            def send_confirmation_email():
                """
                Resend confirmation email.

                Returns:
                Response: Redirect response after sending confirmation email.
                """
                user = current_user
                if not user.email_confirmed:
                    token = self.generate_confirmation_token(user.email)
                    confirm_url = url_for('confirm_email', token=token, _external=True)
                    html = render_template('email/activate.html', confirm_url=confirm_url, SUPPORTED_LANGUAGES=self.languages)
                    self.send_email(user.email, 'Veuillez confirmer votre email', html)
                    self.logger.info(f'Confirmation email resent to: {user.email}')
                    flash(_('Un email de confirmation a été envoyé.'), 'success')
                else:
                    self.logger.info(f'Confirmation email not sent, already confirmed: {user.email}')
                    flash(_('Votre email est déjà confirmé.'), 'info')
                return redirect(url_for('/'))

        @self.route('/logout')
        @login_required
        def logout():
            """
            Handle user logout.

            Returns:
            Response: Redirect response after logging out.
            """
            self.logger.info(f'User logged out: {current_user.username}')
            logout_user()
            return redirect(url_for('index'))


# =================================== Utils ===================================
    def send_verification_email(self, user):
        """
        Send a verification email with a verification code.

        Parameters:
        user (User): The user to send the email to.
        """
        # Generate and send verification code if email is confirmed
        verification_code = Utils.generate_verification_code()
        user.verification_code = verification_code
        self.db.session.commit()
        
        verification_code = Utils.generate_verification_code()
        user.verification_code = verification_code
        user.verification_code_expiry = datetime.now() + timedelta(minutes=10)  # Code valid for 10 minutes
        self.db.session.commit()
        subject = "Votre code de vérification"
        html = render_template('email/verification_code.html',
                            verification_code=verification_code)
        self.send_email(user.email, subject, html)


    def send_email(self, to, subject, template) -> None:
        """
        Send an email.

        Parameters:
        to (str): Recipient email address.
        subject (str): Email subject.
        template (str): HTML template for the email body.
        """
        msg = Message(
            subject,
            recipients=[to],
            html=template,
        )
        self.mail.send(msg)
        # log('INFO', f'Email sent to {to}')


    def generate_confirmation_token(self, email) -> str:
        """
        Generate a confirmation token for email verification.

        Parameters:
        email (str): Email address to generate the token for.

        Returns:
        str: The generated confirmation token.
        """
        serializer = URLSafeTimedSerializer(self.config['SECRET_KEY'])
        return serializer.dumps(email, salt=self.config['SECURITY_PASSWORD_SALT'])


    def confirm_token(self, token: str, expiration=3600):
        """
        Confirm the token for email verification.

        Parameters:
        token (str): The token to confirm.
        expiration (int): Token expiration time in seconds.

        Returns:
        str: The email address associated with the token if valid, else False.
        """
        serializer = URLSafeTimedSerializer(self.config['SECRET_KEY'])
        try:
            email = serializer.loads(
                token,
                salt=self.config['SECURITY_PASSWORD_SALT'],
                max_age=expiration
            )
        except:
            return False
        return email

class Utils:
    """
    Utility class for common utility functions.
    """
    
    @staticmethod
    def generate_verification_code(size: int = 6) -> str:
        """
        Generate a verification code.

        Parameters:
        size (int): Length of the verification code.

        Returns:
        str: The generated verification code.
        """
        CHARS = string.ascii_letters + string.digits
        return ''.join(random.choice(CHARS) for _ in range(size))

    def get_device_info():
        """
        Get device information from the request headers.

        Returns:
        dict: A dictionary containing device information such as user agent, IP address, etc.
        """
        user_agent = request.headers.get('User-Agent')
        ip_address = request.remote_addr
        accept_headers = request.headers.get('Accept')
        platform = request.user_agent.platform
        browser = request.user_agent.browser
        language = request.headers.get('Accept-Language')
        device_info = {
            'user_agent': user_agent,
            'ip_address': ip_address,
            'accept_headers': accept_headers,
            'platform': platform,
            'browser': browser,
            'language': language
        }
        print('Device infos :', *device_info.items(), sep='\n', end='\n\n')
        return device_info
