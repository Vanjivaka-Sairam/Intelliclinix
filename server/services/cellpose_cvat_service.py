import os
from pprint import pprint
import io
import time
import zipfile
import json
from http import HTTPStatus

from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import DataRequest

from flask import current_app


CVAT_HOST = os.getenv('CVAT_API_URL')
CVAT_ADMIN_USERNAME = os.getenv('CVAT_API_USER')
CVAT_ADMIN_PASSWORD = os.getenv('CVAT_API_PASSWORD')


def create_task_from_inference(inference_id: str, task_name: str = None, username: str = None, password: str = None, selected_filenames: list = None):
    """
    Pushes images and masks from an inference result to CVAT using the 3-step flow.
    This function was split out of `cvat_api.py` and houses Cellpose/CVAT upload logic.
    """
    from db import get_db, get_fs
    from bson.objectid import ObjectId

    db = get_db()
    fs = get_fs()

    # Validate and find inference
    try:
        inference_obj_id = ObjectId(inference_id)
    except Exception:
        raise ValueError(f"Invalid inference_id format: {inference_id}")

    inference = db.inferences.find_one({"_id": inference_obj_id})
    if not inference:
        raise ValueError(f"Inference not found: {inference_id}")

    if inference['status'] != 'completed':
        raise ValueError(f"Inference {inference_id} is not completed (status: {inference['status']})")

    if not task_name:
        task_name = f"Inference_{inference_id}"

    image_files = {}
    mask_files = {}

    results = inference.get('results', [])
    if not results:
        raise ValueError(f"Inference {inference_id} has no results")

    for result in results:
        source_filename = result.get('source_filename')
        if not source_filename:
            continue

        if selected_filenames is not None and source_filename not in selected_filenames:
            continue

        source_image_id = result.get('source_image_gridfs_id')
        if source_image_id:
            try:
                image_file = fs.get(ObjectId(source_image_id))
                image_bytes = image_file.read()
                image_files[source_filename] = image_bytes
            except Exception as e:
                current_app.logger.error(f"Failed to read source image {source_image_id}: {e}")
                continue

        class_mask_id = result.get('class_mask_id')
        if class_mask_id:
            try:
                mask_file = fs.get(ObjectId(class_mask_id))
                mask_bytes = mask_file.read()
                mask_files[source_filename] = mask_bytes
            except Exception as e:
                current_app.logger.error(f"Failed to read class mask {class_mask_id}: {e}")

    if not image_files:
        raise ValueError(f"No images found in inference {inference_id}")

    if not mask_files:
        current_app.logger.warning(f"[CVAT] No masks found in inference {inference_id}. Task will be created without annotations.")
        annotation_zip_bytes = None
    else:
        current_app.logger.info(f"[CVAT] Found {len(mask_files)} masks to upload: {list(mask_files.keys())}")
        annotation_zip_bytes = create_cvat_annotation_zip(mask_files)
        current_app.logger.info(f"[CVAT] Created annotation ZIP with {len(mask_files)} mask files, size: {len(annotation_zip_bytes)} bytes")

    config = Configuration(
        host=CVAT_HOST or current_app.config.get('CVAT_API_URL'),
        username=username or CVAT_ADMIN_USERNAME or current_app.config.get('CVAT_ADMIN_USER'),
        password=password or CVAT_ADMIN_PASSWORD or current_app.config.get('CVAT_ADMIN_PASSWORD'),
    )

    with ApiClient(config) as user_client:
        return create_task_from_data(user_client, task_name, image_files, annotation_zip_bytes)


def create_cvat_annotation_zip(mask_files: dict) -> bytes:
    """
    Create a CVAT Segmentation mask 1.1 ZIP. This is identical to the original
    implementation but refactored into its own service file.
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
    NUCLEUS_RGB = (138, 17, 157)

    if not mask_files:
        raise ValueError("No mask files provided for annotation ZIP")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        stems = []

        for filename, mask_obj in mask_files.items():
            stem = os.path.splitext(filename.split('/')[-1].split('\\')[-1])[0]
            stems.append(stem)

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
                mask_int = np.zeros((1, 1), dtype=np.int32)

            if mask_int is None:
                mask_int = np.zeros((1, 1), dtype=np.int32)
            if mask_int.ndim == 3:
                mask_int = mask_int[:, :, 0]

            h, w = mask_int.shape

            class_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            class_rgb[mask_int > 0] = NUCLEUS_RGB

            class_buf = io.BytesIO()
            Image.fromarray(class_rgb).convert('RGB').save(class_buf, format='PNG')
            class_buf.seek(0)

            zf.writestr(f"SegmentationClass/{stem}.png", class_buf.read())

            inst_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            labels = np.unique(mask_int)
            labels = labels[labels > 0]

            for k in labels:
                color = VOC[int(k) % 255 + 1]
                inst_rgb[mask_int == k] = color

            inst_buf = io.BytesIO()
            Image.fromarray(inst_rgb).convert('RGB').save(inst_buf, format='PNG')
            inst_buf.seek(0)

            zf.writestr(f"SegmentationObject/{stem}.png", inst_buf.read())

        default_txt = "".join([f"{s}\n" for s in stems])
        zf.writestr("ImageSets/Segmentation/default.txt", default_txt)

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
    Pushes images and masks to CVAT using the 3-step flow (Task creation, upload images, import annotations).
    """
    tasks_api = user_client.tasks_api

    current_app.logger.info(f"[CVAT] Step 1: Creating Task '{task_name}'...")

    task_spec = {
        'name': task_name,
        'labels': [{
            'name': 'nucleus',
            'color': '#8A119D'
        }]
    }

    try:
        task_data, _ = tasks_api.create(task_spec)
        task_id = task_data.id
        current_app.logger.info(f"[CVAT] Task created with ID: {task_id}")
    except exceptions.ApiException as e:
        current_app.logger.error(f"Exception when creating task: {e}")
        raise ValueError(f"Failed to create CVAT task: {e}")

    current_app.logger.info(f"[CVAT] Step 2: Uploading {len(image_files)} raw images...")

    client_files = []
    for filename, content in image_files.items():
        f = io.BytesIO(content)
        f.name = filename
        client_files.append(f)

    data_request = DataRequest(
        image_quality=70,
        use_zip_chunks=True,
        use_cache=True,
        client_files=client_files
    )

    try:
        (data, response) = tasks_api.create_data(
            id=task_id,
            data_request=data_request,
            _content_type="multipart/form-data",
            _check_status=False,
            _parse_response=False
        )

        if response.status == HTTPStatus.ACCEPTED:
            response_data = json.loads(response.data)
            rq_id = response_data.get("rq_id")
            if rq_id:
                current_app.logger.info(f"[CVAT] Data upload started, request ID: {rq_id}")

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
        elif response.status == HTTPStatus.CREATED:
            current_app.logger.info("[CVAT] Images uploaded and processed immediately.")
        else:
            raise ValueError(f"Unexpected response status: {response.status} - {response.msg}")

    except exceptions.ApiException as e:
        current_app.logger.error(f"Exception when uploading data: {e}")
        raise ValueError(f"Failed to upload images to CVAT: {e}")

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

    current_app.logger.info("[CVAT] Step 3: Importing generated masks...")

    if not annotation_zip_bytes or len(annotation_zip_bytes) == 0:
        current_app.logger.warning("[CVAT] No annotation data to upload, skipping annotation import.")
    else:
        try:
            current_app.logger.info("[CVAT DEBUG] images to upload: %s", list(image_files.keys()))
        except Exception:
            current_app.logger.debug("[CVAT DEBUG] Could not list image_files keys")

        try:
            import zipfile as _zip
            z = _zip.ZipFile(io.BytesIO(annotation_zip_bytes))
            if "ImageSets/Segmentation/default.txt" in z.namelist():
                default_txt = z.read("ImageSets/Segmentation/default.txt").decode('utf-8')
                current_app.logger.info("[CVAT DEBUG] default.txt:\n%s", default_txt)
            else:
                current_app.logger.warning("[CVAT DEBUG] default.txt not found in ZIP")

            current_app.logger.info("[CVAT DEBUG] zip contents: %s", z.namelist())

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

        zip_file_obj = io.BytesIO(annotation_zip_bytes)
        zip_file_obj.name = "masks.zip"

        current_app.logger.info("[CVAT] Importing annotations via dataset import endpoint...")

        try:
            response = None
            used_create_annotations = False

            zip_file_obj.seek(0)

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

            if not used_create_annotations and response:
                try:
                    raw = getattr(response, 'data', None)
                    current_app.logger.debug("[CVAT] dataset import response raw: %s", raw)
                except Exception:
                    pass

                if response.status == HTTPStatus.ACCEPTED:
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
                elif response.status == HTTPStatus.OK:
                    current_app.logger.info("[CVAT] Annotations imported successfully (immediate response).")
                else:
                    error_msg = f"Unexpected response status: {response.status} - {getattr(response, 'msg', '')}"
                    current_app.logger.error(f"[CVAT] {error_msg}")
                    raise ValueError(error_msg)

        except Exception as e:
            current_app.logger.error(f"Error importing annotations: {e}")
            raise ValueError(f"Failed to upload annotations: {e}")

    current_app.logger.info("[CVAT] Workflow complete.")

    host = user_client.configuration.host.replace('/api', '').rstrip('/')
    return {
        "task_id": task_id,
        "task_url": f"{host}/tasks/{task_id}"
    }
