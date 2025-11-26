from flask import Flask, jsonify
from flask_cors import CORS
from config import config
from db import init_db
from blueprints.auth import auth_bp
import commands
from blueprints.inferences import inferences_bp
from blueprints.datasets import datasets_bp
from blueprints.files import files_bp
import os

def create_app(config_name = 'default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    frontend_origin = app.config.get(
        "FRONTEND_ORIGIN",
        os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    )
    CORS(
        app,
        resources={r"/*": {"origins": frontend_origin}},
        supports_credentials=True
    )
    init_db(app)
    commands.register_commands(app)
    
    # we need to register all the blueprints below
    api_prefix = "/api"
    app.register_blueprint(auth_bp, url_prefix=f'{api_prefix}/auth')
    app.register_blueprint(datasets_bp, url_prefix=f'{api_prefix}/datasets')
    app.register_blueprint(inferences_bp, url_prefix=f'{api_prefix}/inferences')
    app.register_blueprint(files_bp, url_prefix=f'{api_prefix}/files')
    
    @app.route('/health')
    def health_check():
        return jsonify({"status" : "ok"})
    
    return app;