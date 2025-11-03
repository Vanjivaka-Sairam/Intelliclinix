from db import get_db, get_fs
from bson.objectid import ObjectId
import datetime
from PIL import Image
import numpy as np
import io
import os
from flask import current_app
from services.storage import save_bytes_to_gridfs
from cellpose import models
from cellpose import io as cp_io

BG_RGB       = (0, 0, 0)
NUCLEUS_RGB = (138, 17, 157) # class color

def voc_colormap(N=256):
    """Standard VOC colormap (unique colors for instances)."""
    cmap = np.zeros((N,3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid & 1) << (7-j))
            g |= (((cid >> 1) & 1) << (7-j))
            b |= (((cid >> 2) & 1) << (7-j))
            cid >>= 3
        cmap[i] = [r, g, b]
    return cmap

VOC_CMAP = voc_colormap(256)

def to_class_rgb(mask_int: np.ndarray) -> np.ndarray:
    """Two-color class mask: BG black, nucleus purple."""
    rgb = np.zeros((*mask_int.shape, 3), np.uint8)
    rgb[mask_int > 0] = NUCLEUS_RGB
    return rgb

def to_instance_rgb(mask_int: np.ndarray) -> np.ndarray:
    """VOC-colored instance mask."""
    h, w = mask_int.shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    labels = np.unique(mask_int)
    labels = labels[labels > 0]
    for k in labels:
        color = VOC_CMAP[int(k) % 255 + 1] # avoid 0 (black)
        out[mask_int == k] = color
    return out

def convert_to_png_bytes(rgb_array: np.ndarray) -> bytes:
    """Converts a numpy RGB array to PNG bytes."""
    img = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
    bytes_io = io.BytesIO()
    img.save(bytes_io, format='PNG')
    return bytes_io.getvalue()


def run_cellpose_model(image_bytes, diameter, channels):
    """
    Runs a specific Cellpose model file on raw image bytes.
    Returns (class_rgb_array, instance_rgb_array)
    """
    
    model_path = os.path.join(current_app.root_path, 'models', 'trained_cellpose_model.pt')
    
    if not os.path.exists(model_path):
        print(f"FATAL: Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading Cellpose model from: {model_path}")

    model = models.CellposeModel(pretrained_model=model_path, gpu=False)
    
    img = cp_io.imread(io.BytesIO(image_bytes))
    
    diam = None if (diameter is None or diameter <= 0) else float(diameter)
    
    print(f"Running model.eval with channels={channels}, diameter={diam}")
    out = model.eval(img, channels=channels, diameter=diam, progress=False)
    
    if isinstance(out, (list, tuple)):
        masks = out[0]
    elif isinstance(out, dict):
        masks = out.get("masks") or out.get("mask")
    else:
        masks = out

    if masks is None:
        masks = np.zeros(img.shape[:2], dtype=np.uint16)
    else:
        masks = masks.astype(np.uint16, copy=False)
    
    print("Cellpose inference complete. Generating CVAT masks.")
    
    class_rgb    = to_class_rgb(masks)
    instance_rgb = to_instance_rgb(masks)
    
    return class_rgb, instance_rgb


def run_inference_job(inference_id_str: str, params: dict):
    """The job logic for a Cellpose inference run."""
    db = get_db()
    fs = get_fs()
    inference_id = ObjectId(inference_id_str)
    
    db.inferences.update_one(
        {"_id": inference_id},
        {"$set": {"status": "running"}}
    )

    inference_doc = db.inferences.find_one({"_id": inference_id})
    dataset_doc = db.datasets.find_one({"_id": inference_doc['dataset_id']})

    diameter = params.get('diameter')
    channels = params.get('channels', [0, 0])

    results = []
    for file_ref in dataset_doc['files']:
        if file_ref['type'] == 'image':
            image_file = fs.get(file_ref['gridfs_id'])
            
            class_rgb_array, instance_rgb_array = run_cellpose_model(
                image_file.read(), 
                diameter=diameter, 
                channels=channels
            )
            
            # 2. Convert both to PNG bytes
            class_mask_bytes = convert_to_png_bytes(class_rgb_array)
            instance_mask_bytes = convert_to_png_bytes(instance_rgb_array)

            # 3. Save both to GridFS
            base_filename = file_ref['filename'].rsplit('.', 1)[0]
            common_metadata = {
                'source_image_gridfs_id': str(file_ref['gridfs_id']),
                'inference_id': str(inference_id)
            }

            class_mask_gridfs_id = save_bytes_to_gridfs(
                class_mask_bytes,
                filename=f"class_{base_filename}.png",
                metadata={**common_metadata, 'type': 'mask_class'}
            )
            
            instance_mask_gridfs_id = save_bytes_to_gridfs(
                instance_mask_bytes,
                filename=f"instance_{base_filename}.png",
                metadata={**common_metadata, 'type': 'mask_instance'}
            )

            # 4. Store both IDs in the results
            results.append({
                "source_filename": file_ref['filename'],
                "class_mask_id": str(class_mask_gridfs_id),
                "instance_mask_id": str(instance_mask_gridfs_id)
            })

    db.inferences.update_one(
        {"_id": inference_id},
        {
            "$set": {
                "status": "completed",
                "finished_at": datetime.datetime.utcnow(),
                "results": results
            }
        }
    )
    print(f"Cellpose inference job {inference_id_str} finished processing.")