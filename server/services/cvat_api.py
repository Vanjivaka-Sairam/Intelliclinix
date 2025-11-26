import os
from pprint import pprint
from flask import Flask, request, jsonify, current_app
from cvat_sdk.api_client import Configuration, ApiClient, exceptions, models
from cvat_sdk.api_client.models import RegisterSerializerExRequest, LoginSerializerExRequest
import io
import time
import zipfile
from http import HTTPStatus
import requests

CVAT_HOST = os.getenv('CVAT_API_URL')
CVAT_ADMIN_USERNAME = os.getenv('CVAT_API_USER') 
CVAT_ADMIN_PASSWORD = os.getenv('CVAT_API_PASSWORD') 

def get_cvat_user_id(username: str):
    """
    Gets the CVAT user ID for a given username.
    
    Args:
        username: The CVAT username to look up
    
    Returns:
        int or None: The user ID if found, None otherwise
    """
    from cvat_sdk.core.helpers import make_client
    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )
    with ApiClient(configuration) as api_client:
        with make_client(current_app.config["CVAT_API_URL"]) as client:
            client.login(
                CVAT_ADMIN_USERNAME,
                CVAT_ADMIN_PASSWORD
            )
            users = client.users.list()
            for user in users:
                if user.username == username:
                    return user.id
    return None

def create_cvat_user(username: str, email: str, password: str, first_name: str, last_name: str):
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    required_fields = ["username", "email", "password", "first_name", "last_name"]
    for f in required_fields:
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400
    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )

    try:
        with ApiClient(configuration) as api_client:
            register_request = RegisterSerializerExRequest(
                username=username,
                email=email,
                password1=password,
                password2=password,
                first_name=first_name,
                last_name=last_name,
            )

            (created_user, response) = api_client.auth_api.create_register(register_request)
            print(created_user)
            print(response)
            return jsonify({"message": "User registered successfully", "data": created_user.to_dict()}), 201

    except exceptions.ApiException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

def cvat_login(host: str, username: str, password: str, email: str = None):

    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )

    with ApiClient(configuration) as api_client:
        login_request = LoginSerializerExRequest(
            username=username,
            email=email or "",
            password=password,
        )

        try:
            (data, response) = api_client.auth_api.create_login(login_request)
            pprint(data)
            return data.to_dict()
        except exceptions.ApiException as e:
            print(f"Exception when calling AuthApi.create_login(): {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error during login: {e}")
            return {"error": str(e)}

def cvat_logout():
    configuration = Configuration(
        host=CVAT_HOST,
        username=CVAT_ADMIN_USERNAME,
        password=CVAT_ADMIN_PASSWORD,
    )

    with ApiClient(configuration) as api_client:
        try:
            (data, response) = api_client.auth_api.create_logout()
            pprint(data)
            return {"message": "User logged out successfully", "data": data.to_dict() if data else {}}
        except exceptions.ApiException as e:
            print(f"Exception when calling AuthApi.create_logout(): {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error during logout: {e}")
            return {"error": str(e)}




def create_task_from_inference(inference_id: str, task_name: str = None, username: str = None, password: str = None, selected_filenames: list = None):
    """
    Pushes images and masks from an inference result to CVAT using the 3-step flow.
    
    Args:
        inference_id: The inference ID to push to CVAT
        task_name: Optional task name (defaults to inference-based name)
        username: Optional CVAT username (defaults to admin)
        password: Optional CVAT password (defaults to admin)
        selected_filenames: Optional list of filenames to include (if None, includes all)
    
    Returns:
        dict: Contains task_id and task_url
    """
    from db import get_db, get_fs
    from bson.objectid import ObjectId
    
    db = get_db()
    fs = get_fs()
    
    # Get inference document
    try:
        inference_obj_id = ObjectId(inference_id)
    except Exception:
        raise ValueError(f"Invalid inference_id format: {inference_id}")
    
    inference = db.inferences.find_one({"_id": inference_obj_id})
    if not inference:
        raise ValueError(f"Inference not found: {inference_id}")
    
    if inference['status'] != 'completed':
        raise ValueError(f"Inference {inference_id} is not completed (status: {inference['status']})")
    
    # Prepare task name
    if not task_name:
        task_name = f"Inference_{inference_id}"
    
    # Collect images and masks from inference results
    image_files = {}  # {filename: file_bytes}
    mask_files = {}   # {filename: mask_bytes} for CVAT annotation ZIP
    
    results = inference.get('results', [])
    if not results:
        raise ValueError(f"Inference {inference_id} has no results")
    
    for result in results:
        source_filename = result.get('source_filename')
        if not source_filename:
            continue
        
        # Filter by selected filenames if provided
        if selected_filenames is not None and source_filename not in selected_filenames:
            continue
        
        # Get source image
        source_image_id = result.get('source_image_gridfs_id')
        if source_image_id:
            try:
                image_file = fs.get(ObjectId(source_image_id))
                image_bytes = image_file.read()
                image_files[source_filename] = image_bytes
            except Exception as e:
                current_app.logger.error(f"Failed to read source image {source_image_id}: {e}")
                continue
        
        # Get class mask for CVAT (CVAT uses class masks for segmentation)
        # The class mask is what CVAT expects for Segmentation mask 1.1 format
        class_mask_id = result.get('class_mask_id')
        if class_mask_id:
            try:
                mask_file = fs.get(ObjectId(class_mask_id))
                mask_bytes = mask_file.read()
                # CVAT expects the mask filename to match the image filename
                mask_files[source_filename] = mask_bytes
            except Exception as e:
                current_app.logger.error(f"Failed to read class mask {class_mask_id}: {e}")
                # Continue without this mask
    
    if not image_files:
        raise ValueError(f"No images found in inference {inference_id}")
    
    if not mask_files:
        current_app.logger.warning(f"[CVAT] No masks found in inference {inference_id}. Task will be created without annotations.")
        annotation_zip_bytes = None
    else:
        current_app.logger.info(f"[CVAT] Found {len(mask_files)} masks to upload: {list(mask_files.keys())}")
        # Create CVAT annotation ZIP in Segmentation mask 1.1 format
        # The format requires masks to be in a ZIP with filenames matching the images
        annotation_zip_bytes = create_cvat_annotation_zip(mask_files)
        current_app.logger.info(f"[CVAT] Created annotation ZIP with {len(mask_files)} mask files, size: {len(annotation_zip_bytes)} bytes")
    
    # Get authenticated client and use it as context manager
    config = Configuration(
        host=CVAT_HOST or current_app.config.get('CVAT_API_URL'),
        username=username or CVAT_ADMIN_USERNAME or current_app.config.get('CVAT_ADMIN_USER'),
        password=password or CVAT_ADMIN_PASSWORD or current_app.config.get('CVAT_ADMIN_PASSWORD'),
    )
    
    # Push to CVAT using the 3-step flow
    with ApiClient(config) as user_client:
        return create_task_from_data(user_client, task_name, image_files, annotation_zip_bytes)


def create_cvat_annotation_zip(mask_files: dict) -> bytes:
    """
    Creates a ZIP file in CVAT Segmentation mask 1.1 format.
    
    CVAT Segmentation mask 1.1 format requires:
    - Masks must be PNG files
    - Mask filenames must exactly match the image filenames
    - Each pixel's RGB value represents a class (background is black 0,0,0)
    
    Args:
        mask_files: Dict of {filename: mask_bytes} where mask_bytes are PNG mask images
    
    Returns:
        bytes: ZIP file bytes
    """
    if not mask_files:
        raise ValueError("No mask files provided for annotation ZIP")
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, mask_bytes in mask_files.items():
            # CVAT expects masks with the same filename as the images
            # The mask should be a PNG where pixel colors represent classes
            # Ensure filename doesn't have path separators
            clean_filename = filename.split('/')[-1].split('\\')[-1]
            zip_file.writestr(clean_filename, mask_bytes)
            current_app.logger.debug(f"Added mask to ZIP: {clean_filename} ({len(mask_bytes)} bytes)")
    
    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()
    current_app.logger.info(f"Created annotation ZIP with {len(mask_files)} files, total size: {len(zip_bytes)} bytes")
    return zip_bytes


def create_task_from_data(user_client: ApiClient, task_name: str, image_files: dict, annotation_zip_bytes: bytes):
    """
    Pushes images and masks to CVAT using the 3-step flow as per CVAT SDK documentation.
    
    Args:
        user_client: Authenticated CVAT client (ApiClient instance)
        task_name: Title for the new task
        image_files: Dict of {filename: file_bytes} for the RAW images
        annotation_zip_bytes: Bytes of the ZIP file containing the masks (Segmentation 1.1 format)
    
    Returns:
        dict: Contains task_id and task_url
    """
    # ------------------------------------------------------------------
    # STEP 1: CREATE TASK
    # Endpoint: POST /api/tasks
    # ------------------------------------------------------------------
    current_app.logger.info(f"[CVAT] Step 1: Creating Task '{task_name}'...")
    
    # Define the label structure (must match your Cellpose classes)
    task_spec = {
        'name': task_name,
        'labels': [{
            'name': 'nucleus',
            'color': '#8A119D'
        }]
    }
    
    try:
        # Use api_client.tasks_api as per documentation
        (task, response) = user_client.tasks_api.create(task_spec)
        task_id = task.id
        current_app.logger.info(f"[CVAT] Task created with ID: {task_id}")
    except exceptions.ApiException as e:
        current_app.logger.error(f"Exception when creating task: {e}")
        raise ValueError(f"Failed to create CVAT task: {e}")

    # ------------------------------------------------------------------
    # STEP 2: UPLOAD DATA (IMAGES)
    # Endpoint: POST /api/tasks/{id}/data/
    # ------------------------------------------------------------------
    current_app.logger.info(f"[CVAT] Step 2: Uploading {len(image_files)} raw images...")
    
    # Convert dict bytes to list of file-like objects with 'name' attribute
    client_files = []
    for filename, content in image_files.items():
        f = io.BytesIO(content)
        f.name = filename  # Critical: CVAT needs the filename to match annotations later
        client_files.append(f)

    # Create data request model
    task_data = models.DataRequest(
        image_quality=70,
        client_files=client_files,
    )

    # Upload data with multipart/form-data content type (required for binary files)
    try:
        (result, response) = user_client.tasks_api.create_data(
            task_id,
            data_request=task_data,
            _content_type="multipart/form-data",
            _check_status=False,
            _parse_response=False
        )
        
        if response.status != HTTPStatus.ACCEPTED:
            raise ValueError(f"Failed to start data upload: {response.status} - {response.msg}")
        
        # Get the request ID from the response
        import json
        response_data = json.loads(response.data)
        rq_id = response_data.get("rq_id")
        
        if not rq_id:
            raise ValueError("No rq_id in response from CVAT data upload")
        
        current_app.logger.info(f"[CVAT] Data upload started, request ID: {rq_id}")
        
    except exceptions.ApiException as e:
        current_app.logger.error(f"Exception when uploading data: {e}")
        raise ValueError(f"Failed to upload images to CVAT: {e}")
    
    # WAIT for server processing using requests_api
    current_app.logger.info("[CVAT] Waiting for images to be processed on server...")
    status = None
    message = None
    
    # Wait up to 5 minutes (300 seconds) for image processing
    max_attempts = 300  # 300 * 1 second = 5 minutes
    for attempt in range(max_attempts):
        try:
            (request_details, response) = user_client.requests_api.retrieve(rq_id)
            status = request_details.status.value
            message = request_details.message
            
            if status in {'finished', 'failed'}:
                break
            time.sleep(1)  # Check every second
        except exceptions.ApiException as e:
            current_app.logger.warning(f"Error checking request status (attempt {attempt + 1}/{max_attempts}): {e}")
            time.sleep(1)
    
    if status != 'finished':
        error_msg = f"CVAT data upload failed with status: {status}"
        if message:
            error_msg += f". Details: {message}"
        raise TimeoutError(error_msg)
    
    current_app.logger.info("[CVAT] Images processed successfully.")
    
    # Update the task object and verify the task size
    try:
        (task, _) = user_client.tasks_api.retrieve(task_id)
        if task.size == 0:
            current_app.logger.warning(f"Task {task_id} has size 0, but upload reported finished")
    except exceptions.ApiException as e:
        current_app.logger.warning(f"Could not verify task size: {e}")

    # ------------------------------------------------------------------
    # STEP 3: UPLOAD ANNOTATIONS (MASKS)
    # Endpoint: POST /api/tasks/{id}/annotations/import
    # Use dataset import endpoint which is the standard way to import annotations
    # ------------------------------------------------------------------
    current_app.logger.info("[CVAT] Step 3: Importing generated masks...")
    
    if not annotation_zip_bytes or len(annotation_zip_bytes) == 0:
        current_app.logger.warning("[CVAT] No annotation data to upload, skipping annotation import.")
    else:
        # Prepare the ZIP file as a file-like object
        zip_file_obj = io.BytesIO(annotation_zip_bytes)
        zip_file_obj.name = "masks.zip"

        # Use REST API directly to import annotations
        # The TasksApi doesn't have create_dataset_import, so we use requests library
        try:
            current_app.logger.info("[CVAT] Starting annotation import via REST API...")
            
            # Get the base URL and authentication headers from ApiClient
            base_url = user_client.configuration.host.rstrip('/')
            url = f"{base_url}/api/tasks/{task_id}/dataset/import"
            
            # Get authentication headers
            headers = user_client.get_common_headers()
            query_params = []
            user_client.update_params_for_auth(headers=headers, queries=query_params, method="POST")
            
            # Prepare multipart/form-data
            zip_file_obj.seek(0)
            files = {
                'annotation_file': ('masks.zip', zip_file_obj, 'application/zip')
            }
            data = {
                'format': 'Segmentation mask 1.1',
                'save_images': 'false'
            }
            
            # Make the request
            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                params=query_params,
                timeout=300
            )
            
            if response.status_code == HTTPStatus.ACCEPTED:
                # Get the request ID from the response
                import json
                response_data = response.json()
                rq_id = response_data.get("rq_id")
                
                if rq_id:
                    current_app.logger.info(f"[CVAT] Annotation import started, request ID: {rq_id}")
                    
                    # Wait for annotation import to complete
                    status = None
                    message = None
                    max_attempts = 300  # Wait up to 5 minutes
                    
                    for attempt in range(max_attempts):
                        try:
                            (request_details, api_response) = user_client.requests_api.retrieve(rq_id)
                            status = request_details.status.value
                            message = request_details.message
                            
                            if status in {'finished', 'failed'}:
                                break
                            time.sleep(1)
                        except exceptions.ApiException as e:
                            current_app.logger.warning(f"Error checking annotation import status (attempt {attempt + 1}/{max_attempts}): {e}")
                            time.sleep(1)
                    
                    if status == 'finished':
                        current_app.logger.info("[CVAT] Annotations imported successfully.")
                    else:
                        error_msg = f"Annotation import failed with status: {status}"
                        if message:
                            error_msg += f". Details: {message}"
                        current_app.logger.error(f"[CVAT] {error_msg}")
                        raise ValueError(error_msg)
                else:
                    current_app.logger.warning("[CVAT] No rq_id in annotation import response, assuming success")
            elif response.status_code == HTTPStatus.OK:
                # Some CVAT versions return 200 OK instead of 202 Accepted
                current_app.logger.info("[CVAT] Annotation import completed immediately (200 OK response).")
            else:
                error_msg = f"Failed to start annotation import: {response.status_code} - {response.text}"
                current_app.logger.error(f"[CVAT] {error_msg}")
                raise ValueError(error_msg)
                
        except Exception as e:
            current_app.logger.error(f"Error using REST API for annotation import: {e}")
            # Try alternative: use create_annotations method (might work for some CVAT versions)
            try:
                current_app.logger.info("[CVAT] Retrying with create_annotations method...")
                zip_file_obj.seek(0)
                user_client.tasks_api.create_annotations(
                    id=task_id,
                    format="Segmentation mask 1.1",
                    annotation_file_request=models.AnnotationFileRequest(
                        annotation_file=zip_file_obj
                    ),
                    _content_type="multipart/form-data"
                )
                current_app.logger.info("[CVAT] Annotations uploaded successfully using create_annotations.")
            except exceptions.ApiException as e2:
                current_app.logger.error(f"Annotation upload failed with create_annotations: {e2}")
                # Last resort: try without content type
                try:
                    current_app.logger.info("[CVAT] Retrying without explicit content type...")
                    zip_file_obj.seek(0)
                    user_client.tasks_api.create_annotations(
                        id=task_id,
                        format="Segmentation mask 1.1",
                        annotation_file_request=models.AnnotationFileRequest(
                            annotation_file=zip_file_obj
                        )
                    )
                    current_app.logger.info("[CVAT] Annotations uploaded successfully.")
                except Exception as e3:
                    current_app.logger.error(f"All annotation upload methods failed: {e3}")
                    raise ValueError(f"Failed to upload annotations. Last error: {e3}")
    
    current_app.logger.info("[CVAT] Workflow complete.")
    
    # Construct the URL for the frontend
    host = user_client.configuration.host.replace('/api', '').rstrip('/')
    return {
        "task_id": task_id,
        "task_url": f"{host}/tasks/{task_id}"
    }
