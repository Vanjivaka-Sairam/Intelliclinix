from flask import Blueprint, jsonify
from utils.security import jwt_required

models_bp = Blueprint("models", __name__)

AVAILABLE_MODELS = [
    {
        "_id": "cellpose_model",
        "name": "Cellpose",
        "runner_name": "cellpose",
        "description": "Cellpose-based segmentation model",
        "input_type": "flat",
    },
    {
        "_id": "ddish_model",
        "name": "D-DISH (Cellpose + YOLO)",
        "runner_name": "d_dish",
        "description": "Hybrid segmentation and classification pipeline",
        "input_type": "flat",
    },
    {
        "_id": "fish_model",
        "name": "FISH (5-Channel)",
        "runner_name": "fish",
        "description": "Requires exactly 5 images: DAPI, FITC, ORANGE, AQUA, SKY",
        "input_type": "grouped_5",
    },
]


_MODELS_BY_ID = {m["_id"]: m for m in AVAILABLE_MODELS}


def get_model_by_id(model_id: str):
    return _MODELS_BY_ID.get(model_id)


@models_bp.route("/", methods=["GET"])
@jwt_required
def list_models(current_user_id):
    return jsonify(AVAILABLE_MODELS), 200
