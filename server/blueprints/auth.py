from flask import Blueprint, request, jsonify, session, current_app
from pydantic import ValidationError
from models.schemas import UserSignup, UserLogin
from server.db import get_db
import cvat_sdk.api_client

from server.services.cvat_api import create_cvat_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods = ['POST'])
def signup():
    db = get_db
    try:
        user_data = UserSignup(**request.json)
    except ValidationError as e:
        return jsonify(e.errors()), 400

    if db.users.find_one({"username": user_data.username}):
        return jsonify({"error": "Username is already taken"}), 409
    if db.users.find_one({"email": user_data.email}):
        return jsonify({"error": "An account with this email already exists"}), 409
    
    try:
        cvat_user = create_cvat_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        if not cvat_user or not cvat_user.id:
            print(f"Failed to create CVAT user for {user_data.username}. Response was: {cvat_user}")
            return jsonify({"error": "Failed to create user in CVAT. The username or email might already exist there."}), 400
    except Exception as e:
        print(f"CRITICAL: Exception while creating CVAT user {user_data.username}: {e}")
        return jsonify({"error": f"An error occurred while contacting the CVAT service: {e}"}), 503

    hashed_password = hash_pass

