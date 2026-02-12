import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'secret_key')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'secret_jwt_key')
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/intelliclinix_db')
    CVAT_API_URL = os.getenv('CVAT_API_URL', 'http://localhost:8080/')
    CVAT_ADMIN_USER = os.getenv('CVAT_ADMIN_USER', 'Vanjivaka_Sairam')
    CVAT_ADMIN_PASSWORD = os.getenv('CVAT_ADMIN_PASSWORD', 'Intelli1@pass')
    
    # Model Paths
    CELLPOSE_MODEL_PATH = os.getenv('CELLPOSE_MODEL_PATH')
    DDISH_CELLPOSE_MODEL_PATH = os.getenv('DDISH_CELLPOSE_MODEL_PATH')
    DDISH_YOLO_MODEL_PATH = os.getenv('DDISH_YOLO_MODEL_PATH')
    
class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development' : DevelopmentConfig,
    'production' : ProductionConfig,
    'default' : DevelopmentConfig
}