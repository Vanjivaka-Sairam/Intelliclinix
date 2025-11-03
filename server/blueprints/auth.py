from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from datetime import datetime
from db import get_db
from schema.schemas import UserSignup, UserLogin
from utils.security import hash_password, verify_password, create_jwt_token, jwt_required
from services.cvat_api import create_cvat_user, cvat_login, cvat_logout
auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/signup", methods=["POST"])
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

    try:
        cvat_response, status_code = create_cvat_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
        )

        if status_code != 201:
            error_message = cvat_response.get_json().get("error", "CVAT registration failed")
            return jsonify({"error": error_message}), 500

        # No need to store cvat_user_id; usernames are unique
    except Exception as e:
        return jsonify({"error": f"CVAT user creation failed: {str(e)}"}), 500

    # Save user to MongoDB only if CVAT registration succeeded
    hashed_password = hash_password(user_data.password)
    user_doc = {
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hashed_password,
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "role": "researcher",
        "created_at": datetime.utcnow()
    }

    user_id = db.users.insert_one(user_doc).inserted_id

    return jsonify({"message": "User registered successfully", "user_id": str(user_id)}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    db = get_db()
    try:
        login_data = UserLogin(**request.json)
    except ValidationError as e:
        return jsonify(e.errors()), 400

    user = db.users.find_one({"username": login_data.username})
    if not user or not verify_password(login_data.password, user["password_hash"]):
        return jsonify({"error": "Invalid username or password"}), 401

    db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}},
    )

    token = create_jwt_token(identity=str(user["_id"]))

    try:
        cvat_login_data = cvat_login(
            host=None,
            username=login_data.username,
            password=login_data.password,
        )
        if "error" in cvat_login_data:
            print(f"Warning: CVAT login failed for {login_data.username}: {cvat_login_data['error']}")
    except Exception as e:
        print(f"Non-critical: CVAT login check failed: {e}")

    return jsonify({"access_token": token}), 200


@auth_bp.route("/logout", methods=["POST"])
@jwt_required
def logout(current_user_id):
    db = get_db()
    user = db.users.find_one({"_id": current_user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
    app_logout_success = True
    cvat_logout_success = False
    cvat_error = None
    
    try:
        cvat_logout_resp = cvat_logout()
        if isinstance(cvat_logout_resp, dict) and cvat_logout_resp.get("error"):
            cvat_error = cvat_logout_resp["error"]
        else:
            cvat_logout_success = True
    except Exception as e:
        cvat_error = str(e)

    ret = {"logout": app_logout_success, "cvat_logout": cvat_logout_success}
    if cvat_error:
        ret["cvat_error"] = cvat_error
    return jsonify(ret), 200
