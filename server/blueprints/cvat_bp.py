from flask import Blueprint, request, jsonify
from db import get_db, get_fs
from utils.security import jwt_required
from bson.objectid import ObjectId
from services.cvat_upload_import import upload_images_to_cvat

cvat_bp = Blueprint('cvat', __name__)

@cvat_bp.route('/inferences/<inference_id>/push_to_cvat', methods=['POST'])
@jwt_required
def push_to_cvat(current_user_id, inference_id):
    db = get_db()
    fs = get_fs()
    data = request.get_json()
    filenames = data.get('filenames', [])
    if not filenames:
        return jsonify({"error": "No filenames provided"}), 400

    inference = db.inferences.find_one({"_id": ObjectId(inference_id)})
    if not inference:
        return jsonify({"error": "Inference not found"}), 404
    if str(inference['requested_by']) != current_user_id:
        return jsonify({"error": "Forbidden"}), 403
    if inference['status'] != 'completed':
        return jsonify({"error": "Inference is not yet complete"}), 400

    try:
        cvat_url = upload_images_to_cvat(fs, inference, filenames)
        return jsonify({"task_url": cvat_url}), 200
    except Exception as e:
        return jsonify({"error": f"CVAT upload failed: {e}"}), 500
