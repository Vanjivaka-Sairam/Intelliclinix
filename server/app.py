from flask import Flask, jsonify
from config import config
from db import init_db
from blueprints.auth import auth_bp
import commands
from blueprints.inferences import inferences_bp
from blueprints.datasets import datasets_bp
from blueprints.files import files_bp

def create_app(config_name = 'default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    init_db(app)
    commands.register_commands(app)
    
    #we need to register all the blueprints below
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(datasets_bp, url_prefix='/datasets')
    app.register_blueprint(inferences_bp, url_prefix='/inferences')
    app.register_blueprint(files_bp, url_prefix='/auth')
    
    @app.route('/health')
    def health_check():
        return jsonify({"status" : "ok"})
    
    return app;