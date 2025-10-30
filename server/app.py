from flask import Flask, jsonify
from config import config
from db import init_db
from blueprints.auth import auth_bp
import commands

def create_app(config_name = 'default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    init_db(app)
    commands.register_commands(app)
    #we need to register all the blueprints below
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    @app.route('/health')
    def health_check():
        return jsonify({"status" : "ok"})
    
    return app;