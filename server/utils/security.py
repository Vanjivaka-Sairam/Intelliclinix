import bcrypt
import jwt

from flask import request, jsonify, current_app
from functools import wraps
import datetime

def hash_password(password : str) -> str:
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_bytes.decode('utf-8')

def verify_password(plain_password : str, hashed_password : str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_jwt_token(identity):
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        'iat': datetime.datetime.utcnow(),
        'sub': identity
    }

    return jwt.encode(
        payload,
        current_app.config['JWT_SECRET_KEY'],
        algorithm='HS256'
    )


def decode_jwt_token(token: str) -> dict:
    """Decode and validate a JWT. Returns the payload dict. Raises jwt exceptions on failure."""
    return jwt.decode(
        token,
        current_app.config['JWT_SECRET_KEY'],
        algorithms=['HS256']
    )

def jwt_required(f):  #we use this to protect routes 
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            payload = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            current_user_id = payload['sub']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(current_user_id=current_user_id, *args, **kwargs)
    return decorated


