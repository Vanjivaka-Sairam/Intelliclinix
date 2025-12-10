from flask import Blueprint, request, jsonify, current_app
from db import get_db
from bson.objectid import ObjectId
from utils.security import jwt_required
import gridfs
from services.cvat_push.cvat_push_manager import CvatService

cvat_bp = Blueprint('cvat', __name__)


@cvat_bp.route('/push-inference/<inference_id>', methods=['POST'])
@jwt_required
def push_inference_to_cvat(current_user_id, inference_id):
    db = get_db()
    fs = gridfs.GridFS(db)

    try:
        inference_obj_id = ObjectId(inference_id)
    except Exception as e:
        return jsonify({
            'error': 'Invalid inference ID'
        }), 400

    inference  = db.inferences.find_one({
        "_id" : inference_obj_id
    })

    if not inference:
        return jsonify({
            'error': 'Inference not found'
        }), 404
    
    if str(inference["requested_by"]) != current_user_id:
        return jsonify({"error" : "forbidden"}), 403

    if inference["status"] != "completed" : 
        return jsonify({"error" : f"Inference not completed (status : {inference['status']})"}), 400
    
    try:
        service = CvatService()
        result = service.push_inference_to_cvat(inference_id, current_user_id)
        return jsonify({
            "message": "Inference pushed to CVAT successfully",
            "data": result
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to push to CVAT: {e}")
        return jsonify({"error": "Failed to push to CVAT"}), 500
