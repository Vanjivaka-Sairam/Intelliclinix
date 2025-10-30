from flask import Blueprint, request, jsonify, session, current_app
from pydantic import ValidationError
from models.schemas import UserSignup, UserLogin
from db import get_db
import cvat_sdk.api_client
from utils.security import hash_password, verify_password,create_jwt_token
from datetime import datetime
from bson.objectid import ObjectId
from services.cvat_api import create_cvat_user

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods = ['POST'])
def signup():
    db = get_db()
    try:
        user_data = UserSignup(**request.json)
    except ValidationError as e:
        return jsonify(e.errors()), 400

    if db.users.find_one({"username": user_data.username}):
        return jsonify({"error": "Username is already taken"}), 409
    if db.users.find_one({"email": user_data.email}):
        return jsonify({"error": "An account with this email already exists"}), 409


## Creating a hashed password 
    hashed_password = hash_password(user_data.password)
## Sving user document to the database
    user_doc = {
        "username": user_data.username,
        "email": user_data.email,
        "password": hashed_password,
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "password_hash": hashed_password,
        "role": "researcher",
        "created_at": datetime.utcnow(),
        "cvat_user_id":None,

    }
    user_id = db.users.insert_one(user_doc).inserted_id
    
    ## Creating the same user on CVAT
    try:
        cvat_user = create_cvat_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        if cvat_user and cvat_user.id:
            db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"cvat_user_id": cvat_user['id']}}
            )
    except Exception as e:
        print(f"Critical: Failed to create corresponding CVAT user for {user_data.username}: {e}")

    return jsonify({"message": "User registered successfully","user_id":str(user_id)}), 201



@auth_bp.route('/login', methods = ['POST'])
def login():
    db = get_db()
    try:
        login_data = UserLogin(**request.json)
    except ValidationError as e:
        return jsonify(e.errors()), 400

    user = db.users.find_one({"username": login_data.username})
    if not user or not verify_password(login_data.password, user['password_hash']):
        return jsonify({"error": "Invalid username or password"}), 401
    db.users.update_one(
        {"_id": user['_id']},
        {"$set": {"last_login": datetime.datetime.utcnow()}}
    )

    token = create_jwt_token(identity=str(user['_id']))

    return jsonify({"access_token": token}), 200

    
    
    

