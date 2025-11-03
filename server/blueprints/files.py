from flask import Blueprint, Response
from db import get_fs
from bson.objectid import ObjectId
from utils.security import jwt_required
import mimetypes

files_bp = Blueprint('files', __name__)

@files_bp.route('/<file_id>')
@jwt_required
def get_gridfs_file(current_user_id, file_id):
    fs = get_fs()
    try:
        gridfs_file = fs.get(ObjectId(file_id))
        
        content_type = mimetypes.guess_type(gridfs_file.filename)[0] or 'application/octet-stream'
        return Response(gridfs_file.read(), mimetype=content_type)
    except Exception as e:
        return Response(f"Error retrieving file: {e}", status=404)