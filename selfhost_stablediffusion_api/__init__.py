from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from .config import Config
from utils import Database
from flask import Flask
import logging

db = Database()

class Server(Flask):
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the Server instance.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.config.from_object(Config)
        self.limiter = Limiter(get_remote_address, default_limits=["200 per day", "50 per hour"])
        self.db = db
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler('gen_serv.log'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)
        self.jwt = JWTManager()
        
        # Import and register blueprints
        from .auth import bp as auth_bp
        self.register_blueprint(auth_bp)

    def run(self, host: str = None, port: int = None, debug: bool = None, load_dotenv: bool = True, **options) -> None:
        self.limiter.init_app(self)
        self.db.init_app(self)
        self.jwt.init_app(self)

        return super().run(host, port, debug, load_dotenv, **options)
