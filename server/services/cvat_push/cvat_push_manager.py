
import os
import io
import time
import json
from http import HTTPStatus
from typing import Dict, Type
from flask import current_app
from bson.objectid import ObjectId
import gridfs
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import DataRequest, AnnotationFileRequest
from db import get_db

from services.cvat_push.cvat_model.cellpose_cvat_push import CellposeModel
from services.cvat_push.cvat_push_base import CvatBase

CVAT_MODEL_REGISTRY: Dict[str, Type[CvatBase]] = {
    "cellpose": CellposeModel,
}

class CvatService:
    def __init__(self):
        self.db = get_db()
        self.fs = gridfs.GridFS(self.db)
        self.cvat_host = current_app.config["CVAT_API_URL"]
        self.cvat_user = current_app.config.get('CVAT_ADMIN_USER')
        self.cvat_pass = current_app.config.get('CVAT_ADMIN_PASSWORD')

    def push_inference_to_cvat(self, inference_id, user_id):
        inference_doc = self.db.inferences.find_one({"_id": ObjectId(inference_id)})
        if not inference_doc:
            raise ValueError("Inference not found")
        
        runner_name = inference_doc.get("runner_name", "cellpose")
        
        model_class = CVAT_MODEL_REGISTRY.get(runner_name)
        if not model_class:
             raise ValueError(f"No CVAT model handler found for runner: {runner_name}")

        model = model_class(self.db, self.fs)
        
        # 1. Load Data
        image_files, image_data_map = model.load_data(inference_id)
        if not image_files:
            raise ValueError("No images found in inference results")

        # 2. Setup Configuration
        configuration = Configuration(
            host=self.cvat_host,
            username=self.cvat_user,
            password=self.cvat_pass,
        )

        with ApiClient(configuration) as client:
            tasks_api = client.tasks_api
            requests_api = client.requests_api
            
            # 3. Create Task
            task_name = f"Inference_{inference_id}"
            task_spec = {
                "name": task_name,
                "labels": model.get_labels(),
            }
            
            try:
                task_data, response = tasks_api.create(task_spec)
                task_id = task_data.id
                current_app.logger.info(f"Task created with ID: {task_id}")
            except exceptions.ApiException as e:
                current_app.logger.error(f"Failed to create task: {e}")
                raise

            # 4. Upload Images
            data_request = DataRequest(
                client_files=image_files,
                image_quality=75
            )

            try:
                result, response = tasks_api.create_data(
                    id=task_id,
                    data_request=data_request,
                    _content_type="multipart/form-data",
                    _check_status=False,
                    _parse_response=False
                )
                
                if response.status == HTTPStatus.ACCEPTED:
                    # Async processing started
                    response_data = json.loads(response.data)
                    rq_id = response_data.get("rq_id")
                    current_app.logger.info(f"Upload started, request ID: {rq_id}")
                    
                    # Poll until completion
                    max_attempts = 300
                    for attempt in range(max_attempts):
                        request_details, _ = requests_api.retrieve(rq_id)
                        status = request_details.status.value
                        message = request_details.message
                        
                        if status == 'finished':
                            current_app.logger.info("Images uploaded successfully")
                            break
                        elif status == 'failed':
                            current_app.logger.error(f"Upload failed: {message}")
                            raise Exception(f"Image upload failed: {message}")
                        
                        time.sleep(1)
                
                elif response.status == HTTPStatus.CREATED:
                    current_app.logger.info("Images uploaded immediately")
                else:
                    raise Exception(f"Unexpected status during image upload: {response.status}")

            except exceptions.ApiException as e:
                 current_app.logger.error(f"Failed to upload images: {e}")
                 # Cleanup task potentially?
                 raise


            zip_buffer = model.prepare_annotations(image_data_map)
            zip_buffer.seek(0)
            zip_buffer.name = "annotations.zip"
            
            annotation_file_request = AnnotationFileRequest(
                annotation_file=zip_buffer
            )

            try:
                tasks_api.create_annotations(
                    id=task_id,
                    format=model.get_annotation_format(),
                    annotation_file_request=annotation_file_request,
                    _content_type="multipart/form-data"
                )
                current_app.logger.info("Successfully uploaded Class and Instance masks.")
            except exceptions.ApiException as e:
                current_app.logger.error(f"Failed to upload annotations: {e}")
                raise

            return {
                "task_id": task_id,
                "url": f"{self.cvat_host}/tasks/{task_id}"
            }
