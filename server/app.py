from flask import Flask, jsonify
from flask_cors import CORS
from config import config
from db import init_db
from blueprints.auth import auth_bp
import commands
from blueprints.inferences import inferences_bp
from blueprints.datasets import datasets_bp
from blueprints.files import files_bp
from blueprints.models import models_bp
from blueprints.cvat_bp import cvat_bp
import os

def create_app(config_name = 'default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Support multiple origins as a comma-separated env var
    # e.g. FRONTEND_ORIGIN=http://localhost:3000,http://192.168.1.5:3000
    raw_origin = app.config.get(
        "FRONTEND_ORIGIN",
        os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    )
    allowed_origins = [o.strip() for o in raw_origin.split(",") if o.strip()]
    if len(allowed_origins) == 1:
        allowed_origins = allowed_origins[0]   # flask-cors prefers a string for single origins

    CORS(
        app,
        resources={r"/*": {"origins": allowed_origins}},
        supports_credentials=True
    )
    init_db(app)
    commands.register_commands(app)
    
    # we need to register all the blueprints below
    api_prefix = "/api"
    app.register_blueprint(auth_bp, url_prefix=f'{api_prefix}/auth')
    app.register_blueprint(datasets_bp, url_prefix=f'{api_prefix}/datasets')
    app.register_blueprint(models_bp, url_prefix=f'{api_prefix}/models')
    app.register_blueprint(inferences_bp, url_prefix=f'{api_prefix}/inferences')
    app.register_blueprint(files_bp, url_prefix=f'{api_prefix}/files')
    app.register_blueprint(cvat_bp, url_prefix=f'{api_prefix}/cvat')
    
    @app.route('/health')
    def health_check():
        return jsonify({"status" : "ok"})
    
    return app;