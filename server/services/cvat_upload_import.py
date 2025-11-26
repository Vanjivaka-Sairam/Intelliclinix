import datetime
from bson.objectid import ObjectId
from flask import current_app
from services.cvat_api import CVAT_HOST, CVAT_ADMIN_USERNAME, CVAT_ADMIN_PASSWORD
from cvat_sdk.core.helpers import make_client

def upload_images_to_cvat(fs, inference, filenames):
    images_to_upload = []
    for result in inference.get('results', []):
        if result['source_filename'] in filenames:
            entry = {'filename': result['source_filename']}
            # Fetch image
            if 'source_image_gridfs_id' in result:
                try:
                    entry['image_bytes'] = fs.get(ObjectId(result['source_image_gridfs_id'])).read()
                except Exception as e:
                    raise RuntimeError(f"Failed to fetch image {result['source_filename']}: {e}")
            # Fetch masks
            if 'class_mask_id' in result and result['class_mask_id']:
                try:
                    entry['class_mask_bytes'] = fs.get(ObjectId(result['class_mask_id'])).read()
                except Exception:
                    entry['class_mask_bytes'] = None
            if 'instance_mask_id' in result and result['instance_mask_id']:
                try:
                    entry['instance_mask_bytes'] = fs.get(ObjectId(result['instance_mask_id'])).read()
                except Exception:
                    entry['instance_mask_bytes'] = None
            images_to_upload.append(entry)

    if not images_to_upload:
        raise RuntimeError("No valid images found for upload")

    task_name = f"Correction_{inference['_id']}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # Try to use the SDK helper if available; otherwise fall back to ApiClient.
    try:
        from cvat_sdk.core.helpers import make_client
        with make_client(CVAT_HOST, CVAT_ADMIN_USERNAME, CVAT_ADMIN_PASSWORD) as client:
            task = client.tasks.create(name=task_name, labels=[{"name": "cell", "attributes": []}])
            for entry in images_to_upload:
                client.tasks.upload_data(task_id=task.id, resources=[(entry['filename'], entry['image_bytes'])])
            cvat_url = f"{CVAT_HOST.rstrip('/')}/tasks/{task.id}"
            return cvat_url
    except Exception:
        # Helper not available (or failed) — attempt to use the generated ApiClient.
        try:
            from cvat_sdk.api_client import Configuration, ApiClient

            configuration = Configuration(host=CVAT_HOST, username=CVAT_ADMIN_USERNAME, password=CVAT_ADMIN_PASSWORD)
            with ApiClient(configuration) as api_client:
                task = None

                # Try common patterns for task creation
                if hasattr(api_client, 'tasks') and hasattr(api_client.tasks, 'create'):
                    task = api_client.tasks.create(name=task_name, labels=[{"name": "cell", "attributes": []}])
                elif hasattr(api_client, 'tasks_api') and hasattr(api_client.tasks_api, 'create'):
                    task = api_client.tasks_api.create(name=task_name, labels=[{"name": "cell", "attributes": []}])
                elif hasattr(api_client, 'tasks_api') and hasattr(api_client.tasks_api, 'create_task'):
                    task = api_client.tasks_api.create_task(name=task_name, labels=[{"name": "cell", "attributes": []}])

                if task is None:
                    raise RuntimeError("Unable to create CVAT task with available SDK methods")

                # Upload images — try a few possible method names on the SDK client
                for entry in images_to_upload:
                    if hasattr(api_client, 'tasks') and hasattr(api_client.tasks, 'upload_data'):
                        api_client.tasks.upload_data(task_id=task.id, resources=[(entry['filename'], entry['image_bytes'])])
                    elif hasattr(api_client, 'tasks_api') and hasattr(api_client.tasks_api, 'create_data'):
                        api_client.tasks_api.create_data(id=task.id, files=[(entry['filename'], entry['image_bytes'])])
                    elif hasattr(api_client, 'tasks_api') and hasattr(api_client.tasks_api, 'upload_data'):
                        api_client.tasks_api.upload_data(id=task.id, files=[(entry['filename'], entry['image_bytes'])])
                    else:
                        raise RuntimeError("Unable to upload data to CVAT task with available SDK methods")

                cvat_id = getattr(task, 'id', getattr(task, 'pk', None))
                if not cvat_id:
                    raise RuntimeError("Failed to determine CVAT task id")
                return f"{CVAT_HOST.rstrip('/')}/tasks/{cvat_id}"
        except Exception as exc:
            raise RuntimeError(f"CVAT upload failed (no suitable client method): {exc}")
