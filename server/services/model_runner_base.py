from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List

from db import get_db, get_fs
from bson.objectid import ObjectId
import datetime


class ModelRunner(ABC):
    """
    Base class for all model runners (strategy interface).

    Each concrete runner is responsible for implementing the actual inference
    logic for a specific model family (e.g. Cellpose, UNet, etc.).
    """

    def __init__(self) -> None:
        self.db = get_db()
        self.fs = get_fs()

    @abstractmethod
    def run_inference_job(self, inference_id_str: str, params: Dict[str, Any]) -> None:
        """
        Run inference for the given inference document.

        Implementations should:
        - Load the inference + dataset metadata from the database
        - Iterate over dataset files
        - Produce masks / outputs
        - Persist outputs and update inference status
        """
        raise NotImplementedError

    # The methods below are helpers inspired by the reference design. They make it
    # easier for runners to update job status in a consistent way.

    def update_inference_status(
        self,
        inference_id: ObjectId,
        status: str,
        results: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Helper: Update inference document status and optional results/error."""
        update_doc: Dict[str, Any] = {
            "$set": {
                "status": status,
                "finished_at": datetime.datetime.utcnow()
                if status in ("completed", "failed")
                else None,
            }
        }

        if results is not None:
            update_doc["$set"]["results"] = results

        if error is not None:
            update_doc["$set"]["notes"] = error

        self.db.inferences.update_one({"_id": inference_id}, update_doc)


