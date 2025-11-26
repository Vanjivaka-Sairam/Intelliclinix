from flask import Blueprint, request, jsonify, current_app
from db import get_db
from bson.objectid import ObjectId
from utils.security import jwt_required
from services.cvat_api import create_task_from_inference

cvat_bp = Blueprint('cvat', __name__)


@cvat_bp.route('/push-inference/<inference_id>', methods=['POST'])
@jwt_required
def push_inference_to_cvat(current_user_id, inference_id):
    """
    Pushes an inference result (images and masks) to CVAT.
    
    The inference must be completed and owned by the current user.
    """
    db = get_db()
    
    # Verify inference exists and belongs to user
    try:
        inference_obj_id = ObjectId(inference_id)
    except Exception:
        return jsonify({"error": "Invalid inference_id format"}), 400
    
    inference = db.inferences.find_one({"_id": inference_obj_id})
    if not inference:
        return jsonify({"error": "Inference not found"}), 404
    
    if str(inference['requested_by']) != current_user_id:
        return jsonify({"error": "Forbidden"}), 403
    
    if inference['status'] != 'completed':
        return jsonify({"error": f"Inference is not completed (status: {inference['status']})"}), 400
    
    # Get optional task name from request
    data = request.get_json() or {}
    task_name = data.get('task_name')
    
    # Optional: Get CVAT credentials from request (if user wants to use different account)
    username = data.get('username')
    password = data.get('password')
    
    try:
        result = create_task_from_inference(
            inference_id=inference_id,
            task_name=task_name,
            username=username,
            password=password
        )
        
        return jsonify({
            "message": "Inference pushed to CVAT successfully",
            "task_id": result["task_id"],
            "task_url": result["task_url"]
        }), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        current_app.logger.error(f"Error pushing inference to CVAT: {e}", exc_info=True)
        return jsonify({"error": f"Failed to push to CVAT: {str(e)}"}), 500

