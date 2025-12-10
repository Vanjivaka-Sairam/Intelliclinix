from flask import Blueprint, jsonify
from utils.security import jwt_required

models_bp = Blueprint("models", __name__)

AVAILABLE_MODELS = [
    {
        "_id": "cellpose_model",
        "name": "Cellpose",
        "runner_name": "cellpose_model",
        "description": "Cellpose-based segmentation model",
    }
]


_MODELS_BY_ID = {m["_id"]: m for m in AVAILABLE_MODELS}


def get_model_by_id(model_id: str):
    return _MODELS_BY_ID.get(model_id)


@models_bp.route("/", methods=["GET"])
@jwt_required
def list_models(current_user_id):
    return jsonify(AVAILABLE_MODELS), 200
