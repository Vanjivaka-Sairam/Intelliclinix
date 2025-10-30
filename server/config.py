import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'secret_key')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'secret_jwt_key')
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/intelliclinix_db')
    CVAT_API_URL = os.getenv('CVAT_API_URL', 'http://localhost:8080/api')
    CVAT_ADMIN_USER = os.getenv('CVAT_ADMIN_USER', 'varshzz')
    CVAT_ADMIN_PASSWORD = os.getenv('CVAT_ADMIN_PASSWORD', '2022mcb1264')
    
class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development' : DevelopmentConfig,
    'production' : ProductionConfig,
    'default' : DevelopmentConfig
}