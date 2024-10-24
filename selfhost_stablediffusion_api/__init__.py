from diffusers import StableDiffusionPipeline, DiffusionPipeline
from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager
from .utils import Database, Utils
from flask_limiter import Limiter
from flask import Flask, jsonify
from .config import Config
import logging
import weakref

db = Database(Config().NO_AUTH)

class GenerationAPI(Flask):
    _pipeline_weakrefs = {}
    
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
        
        from .api import example_bp, txt2img_bp, img2img_bp
        self.register_blueprint(example_bp)
        self.register_blueprint(txt2img_bp)
        self.register_blueprint(img2img_bp)
        
        
        @self.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'models_loaded': dict(GenerationAPI._pipeline_weakrefs)
            })

    def run(self, host: str = None, port: int = None, debug: bool = None, load_dotenv: bool = True, **options) -> None:
        self.limiter.init_app(self)
        self.db.init_app(self)
        self.jwt.init_app(self)

        return super().run(host, port, debug, load_dotenv, **options)

    @classmethod
    def get_pipeline(cls, model_name:str, loader:DiffusionPipeline = StableDiffusionPipeline, **kwargs)->DiffusionPipeline:
        pipe = cls._pipeline_weakrefs() if cls._pipeline_weakrefs.get(model_name, None) else None
        if pipe is None:
            pipe = Utils.load_pipe(model_name=model_name, loader=loader,  **kwargs)
            cls._pipeline_weakrefs[model_name] = weakref.ref(pipe)
        return pipe
