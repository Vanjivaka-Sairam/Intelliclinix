from flask import Blueprint, request, jsonify, current_app
from services.storage import save_file_to_gridfs # Images still use GridFS
from blueprints.models import get_model_by_id
from services.inference_manager import start_managed_inference
from bson.objectid import ObjectId
import json as _json
from db import get_db
import datetime
from bson.objectid import ObjectId
from utils.security import jwt_required 

datasets_bp = Blueprint('datasets', __name__)

@datasets_bp.route('/upload', methods=['POST'])
@jwt_required
def upload_dataset(current_user_id):
    db = get_db()
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    dataset_name = request.form.get('name', 'Untitled Dataset')

    # Model-aware validation
    model_id  = request.form.get('model_id', '')
    model_def = get_model_by_id(model_id) if model_id else None
    input_type = model_def.get("input_type", "flat") if model_def else "flat"

    if input_type == "grouped_5" and len(files) != 5:
        return jsonify({"error": "This model requires exactly 5 images (DAPI, FITC, ORANGE, AQUA, SKY)."}), 400

    file_references = []
    for file in files:
        try:
            gridfs_id = save_file_to_gridfs(
                file,
                metadata={'type': 'image', 'uploader': current_user_id}
            )
            file_references.append({
                "gridfs_id": gridfs_id,
                "filename": file.filename,
                "type": "image"
            })
        except Exception as e:
            return jsonify({"error": f"Failed to save file {file.filename}: {e}"}), 500

    dataset_doc = {
        "name": dataset_name,
        "owner_id": ObjectId(current_user_id),
        "created_at": datetime.datetime.utcnow(),
        "model_id": model_id,
        "input_type": input_type,
        "files": file_references
    }
    dataset_id = db.datasets.insert_one(dataset_doc).inserted_id


    response_payload = {
        "message": "Dataset created successfully",
        "dataset_id": str(dataset_id)
    }

    # Optional: Run inference immediately after upload
    run_inference_flag = request.form.get('run_inference') or request.form.get('run')
    if run_inference_flag and str(run_inference_flag).lower() in {'1', 'true', 'yes', 'on'}:
        # Determine model and params
        model_id = request.form.get('model_id', 'cellpose_default')
        model_def = get_model_by_id(model_id)
        if not model_def:
            return jsonify({"error": f"Unknown model_id '{model_id}'"}), 400

        params_raw = request.form.get('params')
        try:
            params = _json.loads(params_raw) if params_raw else {}
        except Exception:
            return jsonify({"error": "Invalid JSON in params"}), 400

        params.setdefault('diameter', None)
        params.setdefault('channels', [0, 0])

        inference_doc = {
            "dataset_id": ObjectId(dataset_id),
            "requested_by": ObjectId(current_user_id),
            "params": params,
            "model_id": model_id,
            "runner_name": model_def.get('runner_name'),
            "status": "queued",
            "created_at": datetime.datetime.utcnow(),
            "results": []
        }

        inference_id = db.inferences.insert_one(inference_doc).inserted_id

        # Start inference synchronously (same behavior as /inferences/start endpoint)
        try:
            start_managed_inference(str(inference_id), params)
            response_payload['inference_id'] = str(inference_id)
        except Exception as e:
            # update inference doc to failed
            db.inferences.update_one({"_id": inference_id}, {"$set": {"status": "failed", "finished_at": datetime.datetime.utcnow(), "notes": str(e)}})
            current_app.logger.error(f"Failed to run inference created from upload: {e}")
            return jsonify({"error": f"Dataset saved but inference failed to start: {str(e)}"}), 500

    return jsonify(response_payload), 201

@datasets_bp.route('/', methods=['GET'])
@jwt_required
def list_datasets(current_user_id):
    """Lists all datasets owned by the current user."""
    db = get_db()
    datasets = list(db.datasets.find({"owner_id": ObjectId(current_user_id)}))
    
    for ds in datasets:
        ds['_id'] = str(ds['_id'])
        ds['owner_id'] = str(ds['owner_id'])
        for f in ds.get('files', []):
            f['gridfs_id'] = str(f['gridfs_id'])

    return jsonify(datasets), 200