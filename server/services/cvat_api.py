import os
from pprint import pprint
from flask import Flask, request, jsonify, current_app
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import (
    RegisterSerializerExRequest, 
    LoginSerializerExRequest,
    DataRequest
)
import io
import time
import zipfile
import json
from http import HTTPStatus

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
    Create a CVAT Segmentation mask 1.1 ZIP.
    FIXED: Removes non-standard 'image_patches' folder to allow CVAT importer to find files.
    """
    import numpy as np
    from PIL import Image

    def voc_colormap(N=256):
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            cid = i
            for j in range(8):
                r |= ((cid & 1) << (7 - j))
                g |= (((cid >> 1) & 1) << (7 - j))
                b |= (((cid >> 2) & 1) << (7 - j))
                cid >>= 3
            cmap[i] = [r, g, b]
        return cmap

    VOC = voc_colormap(256)
    BG_RGB = (0, 0, 0)
    NUCLEUS_RGB = (138, 17, 157) # Must match labelmap.txt EXACTLY

    if not mask_files:
        raise ValueError("No mask files provided for annotation ZIP")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        stems = []

        for filename, mask_obj in mask_files.items():
            # Compute stem (basename without extension)
            # Example: "data/image_01.jpg" -> "image_01"
            stem = os.path.splitext(filename.split('/')[-1].split('\\')[-1])[0]
            stems.append(stem)

            # --- Mask Processing Logic (Same as your original) ---
            mask_int = None
            if isinstance(mask_obj, bytes):
                try:
                    pil = Image.open(io.BytesIO(mask_obj))
                    if pil.mode in ("I", "I;16") or pil.mode == 'L':
                        mask_int = np.array(pil, dtype=np.int32)
                    else:
                        rgb = pil.convert('RGB')
                        arr = np.array(rgb)
                        diff = np.any(arr != BG_RGB, axis=2)
                        mask_int = np.zeros(diff.shape, dtype=np.int32)
                        mask_int[diff] = 1
                except Exception:
                    mask_int = np.zeros((1, 1), dtype=np.int32)
            elif isinstance(mask_obj, (list, tuple)):
                mask_int = np.array(mask_obj, dtype=np.int32)
            elif isinstance(mask_obj, np.ndarray):
                mask_int = mask_obj.astype(np.int32)
            else:
                # Fallback logic...
                mask_int = np.zeros((1, 1), dtype=np.int32)

            if mask_int is None: 
                mask_int = np.zeros((1, 1), dtype=np.int32)
            if mask_int.ndim == 3:
                mask_int = mask_int[:, :, 0] # Simplify if 3D
            
            h, w = mask_int.shape

            # --- CLASS MASK (SegmentationClass) ---
            class_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            # Ensure strict color matching for the label
            class_rgb[mask_int > 0] = NUCLEUS_RGB

            class_buf = io.BytesIO()
            Image.fromarray(class_rgb).convert('RGB').save(class_buf, format='PNG')
            class_buf.seek(0)

            # CHANGE 1: Write directly to SegmentationClass/{stem}.png
            # Removed "image_patches/"
            zf.writestr(f"SegmentationClass/{stem}.png", class_buf.read())

            # --- INSTANCE MASK (SegmentationObject) ---
            inst_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            labels = np.unique(mask_int)
            labels = labels[labels > 0]

            for k in labels:
                color = VOC[int(k) % 255 + 1]
                inst_rgb[mask_int == k] = color

            inst_buf = io.BytesIO()
            Image.fromarray(inst_rgb).convert('RGB').save(inst_buf, format='PNG')
            inst_buf.seek(0)

            # CHANGE 2: Write directly to SegmentationObject/{stem}.png
            # Removed "image_patches/"
            zf.writestr(f"SegmentationObject/{stem}.png", inst_buf.read())

        # --- Write default.txt ---
        # CHANGE 3: Just list the stems. 
        # Standard VOC/Segmentation 1.1 does not want directory paths here.
        default_txt = "".join([f"{s}\n" for s in stems])
        zf.writestr("ImageSets/Segmentation/default.txt", default_txt)

        # --- Write labelmap.txt ---
        # This part was correct, but ensures colors match NUCLEUS_RGB above
        labelmap_text = (
            "# label:color_rgb:parts:actions\n"
            "background:0,0,0::\n"
            "nucleus:138,17,157::\n"
        )
        zf.writestr("labelmap.txt", labelmap_text)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def create_task_from_data(user_client: ApiClient, task_name: str, image_files: dict, annotation_zip_bytes: bytes):
    """
    Pushes images and masks to CVAT using the 3-step flow described in the Tasks API PDF.
    
    Args:
        user_client: Authenticated CVAT client
        task_name: Title for the new task
        image_files: Dict of {filename: file_bytes} for the RAW images
        annotation_zip_bytes: Bytes of the ZIP file containing the masks (Segmentation 1.1 format)
    
    Returns:
        dict: Contains task_id and task_url
    """
    # Use api_client.tasks_api as shown in temp.md examples
    tasks_api = user_client.tasks_api

    # ------------------------------------------------------------------
    # STEP 1: CREATE TASK (See PDF Page 2)
    # Endpoint: POST /api/tasks
    # ------------------------------------------------------------------
    current_app.logger.info(f"[CVAT] Step 1: Creating Task '{task_name}'...")
    
    # Define the label structure (must match your Cellpose classes)
    # Use dict format as shown in temp.md examples - models can be passed as dicts
    task_spec = {
        'name': task_name,
        'labels': [{
            'name': 'nucleus',
            'color': '#8A119D'
        }]
    }
    
    try:
        # The create method returns a tuple: (data, response)
        # Can accept dict or model objects
        task_data, _ = tasks_api.create(task_spec)
        task_id = task_data.id
        current_app.logger.info(f"[CVAT] Task created with ID: {task_id}")
    except exceptions.ApiException as e:
        current_app.logger.error(f"Exception when creating task: {e}")
        raise ValueError(f"Failed to create CVAT task: {e}")

    # ------------------------------------------------------------------
    # STEP 2: UPLOAD DATA (IMAGES) (See PDF Page 14)
    # Endpoint: POST /api/tasks/{id}/data/
    # ------------------------------------------------------------------
    current_app.logger.info(f"[CVAT] Step 2: Uploading {len(image_files)} raw images...")
    
    # Convert dict bytes to list of file-like objects with 'name' attribute
    client_files = []
    for filename, content in image_files.items():
        f = io.BytesIO(content)
        f.name = filename  # Critical: CVAT needs the filename to match annotations later
        client_files.append(f)

    data_request = DataRequest(
        image_quality=70,
        use_zip_chunks=True,
        use_cache=True,
        client_files=client_files
    )
    
    try:
        # This initiates the upload - returns (data, response) tuple
        # According to temp.md, create_data can return 202 for async operations
        (data, response) = tasks_api.create_data(
            id=task_id, 
            data_request=data_request,
            _content_type="multipart/form-data",
            _check_status=False,
            _parse_response=False
        )
        
        # Check if async (202) or immediate (201)
        if response.status == HTTPStatus.ACCEPTED:  # 202
            # Async upload - need to poll
            response_data = json.loads(response.data)
            rq_id = response_data.get("rq_id")
            
            if rq_id:
                current_app.logger.info(f"[CVAT] Data upload started, request ID: {rq_id}")
                # Wait for upload to complete
                status = None
                message = None
                max_attempts = 300
                
                for attempt in range(max_attempts):
                    try:
                        (request_details, api_response) = user_client.requests_api.retrieve(rq_id)
                        status = request_details.status.value
                        message = request_details.message
                        
                        if status in {'finished', 'failed'}:
                            break
                        time.sleep(1)
                    except exceptions.ApiException as e:
                        current_app.logger.warning(f"Error checking upload status (attempt {attempt + 1}/{max_attempts}): {e}")
                        time.sleep(1)
                
                if status != 'finished':
                    error_msg = f"Data upload failed with status: {status}"
                    if message:
                        error_msg += f". Details: {message}"
                    raise TimeoutError(error_msg)
            else:
                current_app.logger.warning("[CVAT] No rq_id in response, polling task size instead...")
                # Fallback to polling task size
                for attempt in range(30):
                    time.sleep(2)
                    try:
                        task_info, _ = tasks_api.retrieve(id=task_id)
                        if task_info.size > 0:
                            current_app.logger.info("[CVAT] Images processed successfully.")
                            break
                    except exceptions.ApiException as e:
                        current_app.logger.warning(f"Error checking task status (attempt {attempt + 1}/30): {e}")
                        time.sleep(2)
                else:
                    raise TimeoutError("CVAT timed out processing the images.")
        elif response.status == HTTPStatus.CREATED:  # 201
            current_app.logger.info("[CVAT] Images uploaded and processed immediately.")
        else:
            raise ValueError(f"Unexpected response status: {response.status} - {response.msg}")
            
    except exceptions.ApiException as e:
        current_app.logger.error(f"Exception when uploading data: {e}")
        raise ValueError(f"Failed to upload images to CVAT: {e}")
    
    # Verify images are processed by checking task size
    current_app.logger.info("[CVAT] Verifying images are processed...")
    for attempt in range(10):
        time.sleep(1)
        try:
            task_info, _ = tasks_api.retrieve(id=task_id)
            if task_info.size > 0:
                current_app.logger.info(f"[CVAT] Images processed successfully. Task size: {task_info.size}")
                break
        except exceptions.ApiException as e:
            current_app.logger.warning(f"Error checking task status (attempt {attempt + 1}/10): {e}")
            time.sleep(1)

    # ------------------------------------------------------------------
    # STEP 3: UPLOAD ANNOTATIONS (MASKS)
    # Endpoint: POST /api/tasks/{id}/dataset/import
    # Use dataset import endpoint which is the standard way to import annotations
    # ------------------------------------------------------------------
    current_app.logger.info("[CVAT] Step 3: Importing generated masks...")

    if not annotation_zip_bytes or len(annotation_zip_bytes) == 0:
        current_app.logger.warning("[CVAT] No annotation data to upload, skipping annotation import.")
    else:
        # Debug: list image filenames and inspect zip contents
        try:
            current_app.logger.info("[CVAT DEBUG] images to upload: %s", list(image_files.keys()))
        except Exception:
            current_app.logger.debug("[CVAT DEBUG] Could not list image_files keys")

        # Validate the ZIP and log its content before uploading
        try:
            import zipfile as _zip
            z = _zip.ZipFile(io.BytesIO(annotation_zip_bytes))
            if "ImageSets/Segmentation/default.txt" in z.namelist():
                default_txt = z.read("ImageSets/Segmentation/default.txt").decode('utf-8')
                current_app.logger.info("[CVAT DEBUG] default.txt:\n%s", default_txt)
            else:
                current_app.logger.warning("[CVAT DEBUG] default.txt not found in ZIP")

            current_app.logger.info("[CVAT DEBUG] zip contents: %s", z.namelist())

            # Inspect a sample class mask colors
            cls_files = [n for n in z.namelist() if n.startswith("SegmentationClass/")]
            if cls_files:
                try:
                    from PIL import Image as _Image
                    data = z.read(cls_files[0])
                    img = _Image.open(io.BytesIO(data)).convert('RGB')
                    colors = set(tuple(c) for c in list(img.getdata()))
                    current_app.logger.info("[CVAT DEBUG] sample class mask colors (first file): %s", list(colors)[:10])
                except Exception as e:
                    current_app.logger.debug("[CVAT DEBUG] failed to inspect class mask colors: %s", e)
        except Exception as e:
            current_app.logger.warning(f"[CVAT DEBUG] Failed to inspect annotation ZIP before upload: {e}")

        # Prepare the ZIP file object for upload
        zip_file_obj = io.BytesIO(annotation_zip_bytes)
        zip_file_obj.name = "masks.zip"

        current_app.logger.info("[CVAT] Importing annotations via dataset import endpoint...")

        try:
            response = None
            used_create_annotations = False

            # Ensure zip is at the beginning before first attempt
            zip_file_obj.seek(0)

            # Try create_dataset_import first, with two common parameter names
            try:
                if hasattr(tasks_api, 'create_dataset_import'):
                    current_app.logger.info("[CVAT] trying create_dataset_import(dataset_file)...")
                    try:
                        result, response = tasks_api.create_dataset_import(
                            id=task_id,
                            format="Segmentation mask 1.1",
                            dataset_file=zip_file_obj,
                            save_images=False,
                            _content_type="multipart/form-data",
                            _check_status=False,
                            _parse_response=False
                        )
                    except TypeError:
                        # fallback to alternate param name
                        zip_file_obj.seek(0)
                        result, response = tasks_api.create_dataset_import(
                            id=task_id,
                            format="Segmentation mask 1.1",
                            annotation_file=zip_file_obj,
                            save_images=False,
                            _content_type="multipart/form-data",
                            _check_status=False,
                            _parse_response=False
                        )
                else:
                    raise AttributeError("create_dataset_import not available")

            except (AttributeError, exceptions.ApiException, TypeError) as e:
                current_app.logger.info(f"[CVAT] create_dataset_import failed: {e}; trying create_annotations fallback...")
                zip_file_obj.seek(0)
                try:
                    tasks_api.create_annotations(
                        id=task_id,
                        format="Segmentation mask 1.1",
                        annotation_file_request={'annotation_file': zip_file_obj},
                        _content_type="multipart/form-data"
                    )
                    current_app.logger.info("[CVAT] create_annotations succeeded")
                    used_create_annotations = True
                    response = None
                except exceptions.ApiException as e2:
                    current_app.logger.error(f"[CVAT] create_annotations fallback failed: {e2}")
                    raise

            # If we received an async response (create_dataset_import), handle it
            if not used_create_annotations and response:
                # Log raw response data to help debug missing rq_id issues
                try:
                    raw = getattr(response, 'data', None)
                    current_app.logger.debug("[CVAT] dataset import response raw: %s", raw)
                except Exception:
                    pass

                if response.status == HTTPStatus.ACCEPTED:  # 202
                    # Async import - need to poll
                    try:
                        raw = response.data
                        if isinstance(raw, (bytes, bytearray)):
                            raw = raw.decode('utf-8')
                        response_data = json.loads(raw)
                    except Exception:
                        response_data = {}

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
                elif response.status == HTTPStatus.OK:  # 200
                    current_app.logger.info("[CVAT] Annotations imported successfully (immediate response).")
                else:
                    error_msg = f"Unexpected response status: {response.status} - {getattr(response, 'msg', '')}"
                    current_app.logger.error(f"[CVAT] {error_msg}")
                    raise ValueError(error_msg)

        except Exception as e:
            current_app.logger.error(f"Error importing annotations: {e}")
            raise ValueError(f"Failed to upload annotations: {e}")
    
    current_app.logger.info("[CVAT] Workflow complete.")
    
    # Construct the URL for the frontend
    host = user_client.configuration.host.replace('/api', '').rstrip('/')
    return {
        "task_id": task_id,
        "task_url": f"{host}/tasks/{task_id}"
    }
