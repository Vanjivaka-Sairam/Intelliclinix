from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List

from db import get_db, get_fs
from bson.objectid import ObjectId
import datetime


class ModelRunner(ABC):

    def __init__(self) -> None:
        self.db = get_db()
        self.fs = get_fs()

    @abstractmethod
    def run_inference_job(self, inference_id_str: str) -> None:

        raise NotImplementedError

    def update_inference_status(
        self,
        inference_id: ObjectId,
        status: str,
        results: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> None:
    
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


