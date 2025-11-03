from flask import Blueprint, request, jsonify
from services.storage import save_file_to_gridfs # Images still use GridFS
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
    
    file_references = []
    for file in files:
        try:
            # Images are saved to GridFS
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
        "files": file_references
    }
    dataset_id = db.datasets.insert_one(dataset_doc).inserted_id

    return jsonify({
        "message": "Dataset created successfully",
        "dataset_id": str(dataset_id)
    }), 201

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