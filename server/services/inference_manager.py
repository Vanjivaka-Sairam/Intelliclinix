from typing import Dict, Type

from bson.objectid import ObjectId
from flask import current_app
import datetime

from db import get_db
from services.cellpose_runner import CellposeRunner
from services.model_runner_base import ModelRunner


# Registry mapping logical runner names to concrete runner classes.
# Additional runners can be added here without changing the API layer.
RUNNER_REGISTRY: Dict[str, Type[ModelRunner]] = {
    "cellpose": CellposeRunner,
}


def start_managed_inference(inference_id_str: str, params: dict) -> None:
    """
    Dispatcher/context that selects and executes the appropriate model runner.

    How it works:
    1. Load the inference document by ID
    2. Read the stored runner_name from the document
    3. Look up the corresponding runner class in RUNNER_REGISTRY
    4. Instantiate the runner and invoke run_inference_job()
    """
    db = get_db()
    inference_id = ObjectId(inference_id_str)

    try:
        current_app.logger.info(f"[DISPATCHER] Starting inference: {inference_id_str}")

        inference_doc = db.inferences.find_one({"_id": inference_id})
        if not inference_doc:
            raise ValueError("Inference job not found")

        runner_name = inference_doc.get("runner_name")
        if not runner_name:
            raise ValueError("Inference document is missing 'runner_name'")

        if runner_name not in RUNNER_REGISTRY:
            raise NotImplementedError(
                f"Runner '{runner_name}' is not implemented. "
                f"Available runners: {list(RUNNER_REGISTRY.keys())}"
            )

        runner_class = RUNNER_REGISTRY[runner_name]
        current_app.logger.info(
            f"[DISPATCHER] Selected runner class: {runner_class.__name__}"
        )

        runner = runner_class()
        current_app.logger.info(
            f"[DISPATCHER] Invoking run_inference_job on {runner_class.__name__}"
        )
        runner.run_inference_job(inference_id_str, params)

        current_app.logger.info(
            f"[DISPATCHER] Inference {inference_id_str} completed successfully"
        )

    except Exception as e:
        current_app.logger.error(
            f"[DISPATCHER] Inference {inference_id_str} failed: {e}"
        )
        db.inferences.update_one(
            {"_id": inference_id},
            {
                "$set": {
                    "status": "failed",
                    "finished_at": datetime.datetime.utcnow(),
                    "notes": str(e),
                }
            },
        )
        raise


