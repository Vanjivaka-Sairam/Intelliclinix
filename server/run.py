import os
from dotenv import load_dotenv

load_dotenv()
from app import create_app

config_name = os.getenv('FLASK_CONFIG', 'default')
app = create_app(config_name)

if __name__ == '__main__':
    app.run(debug = True, port = 5001)