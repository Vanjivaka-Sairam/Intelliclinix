from flask import Blueprint, jsonify
from utils.security import jwt_required

models_bp = Blueprint("models", __name__)

AVAILABLE_MODELS = [
    {
        "_id": "cellpose_default",
        "name": "Cellpose Nuclei",
        "runner_name": "cellpose",
        "description": "General-purpose cell/nuclei segmentation model",
    },
    {
        "_id": "cellpose_cytoplasm",
        "name": "Cellpose Cytoplasm",
        "runner_name": "cellpose",
        "description": "Cellpose model tuned for cytoplasm segmentation",
    },
]


@models_bp.route("/", methods=["GET"])
@jwt_required
def list_models(current_user_id):
    return jsonify(AVAILABLE_MODELS), 200


