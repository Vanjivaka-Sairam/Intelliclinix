from flask import Blueprint, request, jsonify, Response, current_app
from db import get_db
import datetime
from bson.objectid import ObjectId
from utils.security import jwt_required
from services.cellpose_runner import run_inference_job
from db import get_db, get_fs
import io
import zipfile
import os

inferences_bp = Blueprint('inferences', __name__)

@inferences_bp.route('/start', methods=['POST'])
@jwt_required
def start_inference(current_user_id):
    """Starts a new inference job using the hardcoded Cellpose runner."""
    db = get_db()
    data = request.json
    
    dataset_id = data.get('dataset_id')
    # Get Cellpose params from request, with defaults
    params = data.get('params', {})
    params.setdefault('diameter', None)
    params.setdefault('channels', [0, 0])

    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400

    inference_doc = {
        "dataset_id": ObjectId(dataset_id),
        "requested_by": ObjectId(current_user_id),
        "params": params, 
        "status": "queued",
        "created_at": datetime.datetime.utcnow(),
        "results": []
    }
    inference_id = db.inferences.insert_one(inference_doc).inserted_id

    try:
        run_inference_job(str(inference_id), params)
        
    except Exception as e:
        db.inferences.update_one(
            {"_id": inference_id},
            {"$set": {"status": "failed", "finished_at": datetime.datetime.utcnow(), "notes": str(e)}}
        )
        print(f"Inference {inference_id} failed:")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}"}), 500

    return jsonify({
        "message": "Inference job completed", 
        "inference_id": str(inference_id)
    }), 200


@inferences_bp.route('/<inference_id>', methods=['GET'])
@jwt_required
def get_inference_status(current_user_id, inference_id):
    """Retrieves the status and results of an inference job."""
    db = get_db()
    inference = db.inferences.find_one({"_id": ObjectId(inference_id)})
    if not inference:
        return jsonify({"error": "Inference not found"}), 404
    
    if str(inference['requested_by']) != current_user_id:
        return jsonify({"error": "Forbidden"}), 403

    inference['_id'] = str(inference['_id'])
    inference['dataset_id'] = str(inference['dataset_id'])
    inference['requested_by'] = str(inference['requested_by'])
    
    # Serializing the new mask IDs for the frontend
    for res in inference.get('results', []):
        if 'class_mask_id' in res:
            res['class_mask_id'] = str(res['class_mask_id'])
        if 'instance_mask_id' in res:
            res['instance_mask_id'] = str(res['instance_mask_id'])
    
    return jsonify(inference), 200

@inferences_bp.route('/<inference_id>/download', methods=['GET'])
@jwt_required
def download_inference_zip(current_user_id, inference_id): #downloads all mask results for an inference as a ZIP file.
    db = get_db()
    fs = get_fs()
    
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
        return jsonify({"error": "Inference is not yet complete"}), 400
    
    # Create an in-memory ZIP file
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        for result in inference.get('results', []):
            source_filename = result['source_filename']
            base_name = os.path.splitext(source_filename)[0] 

            # Add Class Mask
            if 'class_mask_id' in result:
                try:
                    file_data = fs.get(ObjectId(result['class_mask_id'])).read()
                    zip_path = f"{base_name}_class_mask.png"
                    zf.writestr(zip_path, file_data)
                except Exception as e:
                    current_app.logger.error(f"Failed to read class_mask {result['class_mask_id']}: {e}")

            # Add Instance Mask
            if 'instance_mask_id' in result:
                try:
                    file_data = fs.get(ObjectId(result['instance_mask_id'])).read()
                    zip_path = f"{base_name}_instance_mask.png"
                    zf.writestr(zip_path, file_data)
                except Exception as e:
                    current_app.logger.error(f"Failed to read instance_mask {result['instance_mask_id']}: {e}")

    memory_file.seek(0)
    
    return Response(
        memory_file,
        mimetype='application/zip',
        headers={'Content-Disposition': f'attachment;filename=inference_{inference_id}.zip'}
    )