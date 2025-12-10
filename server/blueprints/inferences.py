from flask import Blueprint, request, jsonify, Response, current_app
from db import get_db
import datetime
from bson.objectid import ObjectId
from utils.security import jwt_required
from db import get_db, get_fs
from blueprints.models import get_model_by_id
from services.inference_manager import start_managed_inference
import io
import zipfile
import os

inferences_bp = Blueprint('inferences', __name__)


@inferences_bp.route('/<inference_id>/classify', methods=['POST'])
@jwt_required
def classify_inference(current_user_id, inference_id):
    """Apply classification labels to an inference (either top-level or per-result).

    Accepts JSON body with either:
      - { "classification": {"label": ..., "confidence": ...} }
      - { "result_classifications": [{"source_filename": ..., "label": ..., "confidence": ...}, ...] }
    """
    db = get_db()
    data = request.get_json() or {}

    try:
        inference_obj_id = ObjectId(inference_id)
    except Exception:
        return jsonify({"error": "Invalid inference_id format"}), 400

    inference = db.inferences.find_one({"_id": inference_obj_id})
    if not inference:
        return jsonify({"error": "Inference not found"}), 404

    if str(inference['requested_by']) != current_user_id:
        return jsonify({"error": "Forbidden"}), 403

    # Option 1: top-level classification
    if 'classification' in data:
        db.inferences.update_one({"_id": inference_obj_id}, {"$set": {"classification": data['classification']}})
        return jsonify({"message": "Inference classified"}), 200

    # Option 2: per-result classifications
    if 'result_classifications' in data and isinstance(data['result_classifications'], list):
        updates = data['result_classifications']
        # Load current results and update entries
        inference = db.inferences.find_one({"_id": inference_obj_id})
        results = inference.get('results', [])
        # For each provided classification, attempt to match by source_filename
        for c in updates:
            filename = c.get('source_filename')
            label = c.get('label')
            confidence = c.get('confidence')
            for r in results:
                if r.get('source_filename') == filename:
                    r['classification'] = {"label": label, "confidence": confidence}
        db.inferences.update_one({"_id": inference_obj_id}, {"$set": {"results": results}})
        return jsonify({"message": "Result classifications updated"}), 200

    return jsonify({"error": "No classification data provided"}), 400


@inferences_bp.route('/<inference_id>', methods=['DELETE'])
@jwt_required
def delete_inference(current_user_id, inference_id):
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

    # Attempt to remove any GridFS artifacts referenced in results
    for res in inference.get('results', []):
        artifacts = res.get('artifacts', [])
        for a in artifacts:
            gf = a.get('gridfs_id')
            if gf:
                try:
                    fs.delete(ObjectId(gf))
                except Exception:
                    # log and continue; failure to delete a file should not block removal of the record
                    current_app.logger.warning(f"Failed to delete gridfs file {gf} while deleting inference {inference_id}")

    # Delete the inference document
    db.inferences.delete_one({"_id": inference_obj_id})

    return jsonify({"message": "Inference deleted"}), 200


@inferences_bp.route('/archive', methods=['POST'])
@jwt_required
def bulk_archive_inferences(current_user_id):
    """Archive or unarchive a set of inference IDs.

    Body: { "ids": ["id1","id2"], "archived": true }
    """
    db = get_db()
    data = request.get_json() or {}
    ids = data.get('ids')
    archived = bool(data.get('archived', True))

    if not ids or not isinstance(ids, list):
        return jsonify({"error": "ids must be a list of inference ids"}), 400

    object_ids = []
    for i in ids:
        try:
            object_ids.append(ObjectId(i))
        except Exception:
            return jsonify({"error": f"Invalid inference id: {i}"}), 400

    result = db.inferences.update_many({"_id": {"$in": object_ids}, "requested_by": ObjectId(current_user_id)}, {"$set": {"archived": archived}})
    return jsonify({"matched": result.matched_count, "modified": result.modified_count}), 200




@inferences_bp.route('/start', methods=['POST'])
@jwt_required
def start_inference(current_user_id):

    db = get_db()
    data = request.json

    dataset_id = data.get('dataset_id')
    model_id = data.get('model_id', 'cellpose_model') #setting cellpose as a default model
    model_def = get_model_by_id(model_id)
    if not model_def:
        return jsonify({"error": f"Unknown model_id '{model_id}'"}), 400

    runner_name = model_def["runner_name"]
    params = data.get('params', {}) or {}

    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400

    inference_doc = {
        "dataset_id": ObjectId(dataset_id),
        "requested_by": ObjectId(current_user_id),
        "params": params,
        "model_id": model_id,
        "runner_name": runner_name,
        "status": "queued",
        "created_at": datetime.datetime.utcnow(),
        "results": []
    }
    inference_id = db.inferences.insert_one(inference_doc).inserted_id

    try:
        start_managed_inference(str(inference_id))

    except Exception as e:
        db.inferences.update_one(
            {"_id": inference_id},
            {"$set": {
                "status": "failed",
                "finished_at": datetime.datetime.utcnow(),
                "notes": str(e),
            }}
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
    
    # Serializing mask / artifact IDs for the frontend
    for res in inference.get('results', []):
        if 'class_mask_id' in res and res['class_mask_id'] is not None:
            res['class_mask_id'] = str(res['class_mask_id'])
        if 'instance_mask_id' in res and res['instance_mask_id'] is not None:
            res['instance_mask_id'] = str(res['instance_mask_id'])

        # New generic artifacts array: ensure any gridfs_id values are strings.
        artifacts = res.get("artifacts", [])
        for artifact in artifacts:
            if "gridfs_id" in artifact:
                artifact["gridfs_id"] = str(artifact["gridfs_id"])
    
    return jsonify(inference), 200

@inferences_bp.route('/<inference_id>/download', methods=['GET'])
@jwt_required
def download_inference_zip(current_user_id, inference_id):
    """
    Downloads all mask results for an inference as a ZIP file.
    The ZIP contains a folder for each dataset file, containing
    the image and its masks.
    """
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
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        for result in inference.get('results', []):
            source_filename = result['source_filename']
            # Create a folder name based on the file stem (e.g., 'image_01')
            folder_name = os.path.splitext(source_filename)[0]

            # 1. Add Original Image
            # Path inside zip: image_01/image_01.png
            if 'source_image_gridfs_id' in result:
                try:
                    file_data = fs.get(ObjectId(result['source_image_gridfs_id'])).read()
                    zip_path = os.path.join(folder_name, source_filename)
                    zf.writestr(zip_path, file_data)
                except Exception as e:
                    current_app.logger.error(f"Failed to read source_image {result['source_image_gridfs_id']}: {e}")

            # 2. Add artifacts (generic, supports multiple models)
            artifacts = result.get("artifacts", [])
            if artifacts:
                # Preferred path: use the generic artifacts list (supports any model).
                for artifact in artifacts:
                    gridfs_id = artifact.get("gridfs_id")
                    if not gridfs_id:
                        continue
                    try:
                        file_data = fs.get(ObjectId(gridfs_id)).read()
                        # If the artifact provides its own filename, use it; otherwise
                        # derive a simple name based on kind.
                        artifact_filename = artifact.get(
                            "filename",
                            f"{folder_name}_{artifact.get('kind', 'artifact')}.bin",
                        )
                        zip_path = os.path.join(folder_name, artifact_filename)
                        zf.writestr(zip_path, file_data)
                    except Exception as e:
                        current_app.logger.error(
                            f"Failed to read artifact {gridfs_id} (kind={artifact.get('kind')}): {e}"
                        )
            else:
                # Backwards-compatible path: fall back to class_mask_id / instance_mask_id
                # if no artifacts list is present (older jobs).
                # 2a. Add Class Mask
                if 'class_mask_id' in result and result['class_mask_id']:
                    try:
                        file_data = fs.get(ObjectId(result['class_mask_id'])).read()
                        zip_path = os.path.join(folder_name, f"{folder_name}_class_mask.png")
                        zf.writestr(zip_path, file_data)
                    except Exception as e:
                        current_app.logger.error(f"Failed to read class_mask {result['class_mask_id']}: {e}")

                # 2b. Add Instance Mask
                if 'instance_mask_id' in result and result['instance_mask_id']:
                    try:
                        file_data = fs.get(ObjectId(result['instance_mask_id'])).read()
                        zip_path = os.path.join(folder_name, f"{folder_name}_instance_mask.png")
                        zf.writestr(zip_path, file_data)
                    except Exception as e:
                        current_app.logger.error(f"Failed to read instance_mask {result['instance_mask_id']}: {e}")

    memory_file.seek(0)
    
    return Response(
        memory_file,
        mimetype='application/zip',
        headers={'Content-Disposition': f'attachment;filename=inference_{inference_id}.zip'}
    )


@inferences_bp.route('/', methods=['GET'])
@jwt_required
def list_inferences(current_user_id):
    db = get_db()
    try:
        requester_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({"error": "Invalid user id"}), 400

    # By default, do not return archived inferences unless explicitly requested
    query = {"requested_by": requester_id, "archived": {"$ne": True}}
    requested_dataset = request.args.get("dataset_id")
    if requested_dataset:
        try:
            query["dataset_id"] = ObjectId(requested_dataset)
        except Exception:
            return jsonify({"error": "Invalid dataset_id"}), 400
    if status := request.args.get("status"):
        query["status"] = status
    # Optional: filter by model id (e.g., ?model_id=cellpose_default)
    if model_id := request.args.get("model_id"):
        query["model_id"] = model_id
    if archived := request.args.get("archived"):
        if archived.lower() in {'1', 'true', 'yes', 'on'}:
            query["archived"] = True
        elif archived.lower() in {'0', 'false', 'no', 'off'}:
            query["archived"] = {"$ne": True}

    records = list(db.inferences.find(query).sort("created_at", -1))
    for record in records:
        record["_id"] = str(record["_id"])
        record["dataset_id"] = str(record["dataset_id"])
        record["requested_by"] = str(record["requested_by"])
        for result in record.get("results", []):
            # Backwards-compatible: stringify legacy mask IDs if present
            if "class_mask_id" in result and result["class_mask_id"] is not None:
                result["class_mask_id"] = str(result["class_mask_id"])
            if "instance_mask_id" in result and result["instance_mask_id"] is not None:
                result["instance_mask_id"] = str(result["instance_mask_id"])

            # Generic artifacts: ensure any gridfs_id values are strings
            artifacts = result.get("artifacts", [])
            for artifact in artifacts:
                if "gridfs_id" in artifact:
                    artifact["gridfs_id"] = str(artifact["gridfs_id"])

    return jsonify(records), 200
