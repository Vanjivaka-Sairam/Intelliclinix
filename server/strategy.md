# Strategy Pattern & Dynamic Dispatch Implementation Guide

## What is Strategy Pattern?

**Strategy Pattern** is a behavioral design pattern that:
- Defines a family of algorithms (different inference scripts)
- Encapsulates each one in a separate class (runners)
- Makes them interchangeable at runtime
- Lets the client choose which one to use (based on user's model selection)

**In your case**: User selects Model1 or Model2 → Backend automatically routes to the correct inference script without any frontend changes.

---

## Complete Implementation Steps

### Step 1: Base Class (Already Provided)

**File: `backend/services/model_runner_base.py`**

```python
from abc import ABC, abstractmethod
from db import get_db, get_fs
from bson.objectid import ObjectId
import datetime

class ModelRunner(ABC):
    """Base class for all model runners - defines the strategy interface."""
    
    def __init__(self):
        self.db = get_db()
        self.fs = get_fs()
    
    @abstractmethod
    def run_inference_job(self, inference_id_str: str) -> None:
        """Run inference on images in the dataset."""
        pass
    
    @abstractmethod
    def retrain_job(self, annotation_document_id_str: str, model_id_str: str) -> None:
        """Retrain the model with annotated data."""
        pass
    
    def update_inference_status(self, inference_id: ObjectId, status: str, results=None, error=None):
        """Helper: Update inference document status."""
        update_doc = {
            "$set": {
                "status": status,
                "finished_at": datetime.datetime.utcnow() if status in ["completed", "failed"] else None
            }
        }
        
        if results:
            update_doc["$set"]["results"] = results
        
        if error:
            update_doc["$set"]["notes"] = error
        
        self.db.inferences.update_one({"_id": inference_id}, update_doc)
    
    def update_retraining_status(self, job_id: ObjectId, status: str, logs=None, error=None, new_model_id=None):
        """Helper: Update retraining job status."""
        update_doc = {
            "$set": {
                "status": status,
                "finished_at": datetime.datetime.utcnow() if status in ["completed", "failed"] else None
            }
        }
        
        if error:
            update_doc["$set"]["error"] = error
        
        if new_model_id:
            update_doc["$set"]["new_model_id"] = new_model_id
        
        if logs:
            update_doc["$push"] = {"logs": {"$each": logs}}
        
        self.db.retraining_jobs.update_one({"_id": job_id}, update_doc)
```

**Purpose**: Defines the contract (interface) that all model runners must implement.

---

### Step 2: Create Model1 Runner (Concrete Strategy 1)

**File: `backend/services/runners/model1_runner.py`**

```python
from services.model_runner_base import ModelRunner
from db import get_db, get_fs
from bson.objectid import ObjectId
import datetime
from PIL import Image
import numpy as np
import io
from services.storage import save_bytes_to_gridfs
from flask import current_app
import torch

class Model1Runner(ModelRunner):
    """Concrete Strategy 1: Model1 with specific preprocessing and inference logic."""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def load_model(self, model_path):
        """Load Model1 from file."""
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        return self.model
    
    def preprocess_image(self, image_bytes):
        """
        Model1-specific preprocessing.
        
        Example: Model1 expects:
        - Grayscale images
        - 512x512 resolution
        - Normalized to [0, 1]
        """
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        image = image.resize((512, 512))
        
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def run_model1_inference(self, image_tensor):
        """Model1-specific inference logic."""
        with torch.no_grad():
            output = self.model(image_tensor)
            mask = torch.sigmoid(output)
            mask = (mask > 0.5).float()
            mask_array = mask.cpu().numpy()[0, 0]
        
        return (mask_array * 255).astype(np.uint8)
    
    def run_inference_job(self, inference_id_str: str):
        """
        Execute Model1 inference.
        This method is called by the dispatcher when user selects Model1.
        """
        inference_id = ObjectId(inference_id_str)
        
        try:
            self.update_inference_status(inference_id, "running")
            current_app.logger.info(f"[Model1] Starting inference: {inference_id_str}")
            
            # Get documents from database
            inference_doc = self.db.inferences.find_one({"_id": inference_id})
            dataset_doc = self.db.datasets.find_one({"_id": inference_doc['dataset_id']})
            model_doc = self.db.models.find_one({"_id": inference_doc['model_id']})
            
            # Load model from GridFS
            model_file = self.fs.get(model_doc['artifact_id'])
            model_path = f"/tmp/model1_{inference_id}.pth"
            with open(model_path, 'wb') as f:
                f.write(model_file.read())
            
            self.load_model(model_path)
            
            results = []
            
            # Process each image in dataset
            for file_ref in dataset_doc.get('files', []):
                if file_ref['type'] == 'image':
                    try:
                        # Load image
                        image_file = self.fs.get(file_ref['gridfs_id'])
                        image_bytes = image_file.read()
                        
                        # Preprocess (Model1-specific)
                        image_tensor = self.preprocess_image(image_bytes)
                        
                        # Run inference (Model1-specific)
                        mask_array = self.run_model1_inference(image_tensor)
                        
                        # Save result
                        mask_img = Image.fromarray(mask_array)
                        mask_bytes_io = io.BytesIO()
                        mask_img.save(mask_bytes_io, format='PNG')
                        mask_bytes = mask_bytes_io.getvalue()
                        
                        mask_filename = f"mask_{file_ref['filename'].rsplit('.', 1)[0]}.png"
                        mask_gridfs_id = save_bytes_to_gridfs(
                            mask_bytes,
                            filename=mask_filename,
                            metadata={
                                'type': 'mask',
                                'model': 'model1',
                                'source_image_gridfs_id': str(file_ref['gridfs_id']),
                                'inference_id': str(inference_id)
                            }
                        )
                        
                        results.append({
                            "source_filename": file_ref['filename'],
                            "source_image_gridfs_id": str(file_ref['gridfs_id']),
                            "class_mask_id": str(mask_gridfs_id),
                            "instance_mask_id": None
                        })
                        
                        current_app.logger.info(f"[Model1] Processed: {file_ref['filename']}")
                    
                    except Exception as e:
                        current_app.logger.error(f"[Model1] Error processing {file_ref['filename']}: {e}")
                        continue
            
            self.update_inference_status(inference_id, "completed", results=results)
            current_app.logger.info(f"[Model1] Inference completed with {len(results)} results")
        
        except Exception as e:
            current_app.logger.error(f"[Model1] Inference failed: {e}")
            self.update_inference_status(inference_id, "failed", error=str(e))
    
    def retrain_job(self, annotation_document_id_str: str, model_id_str: str):
        """Model1-specific retraining logic."""
        current_app.logger.info(f"[Model1] Starting retraining...")
        # Implement Model1 retraining
        pass
```

**Key Points**:
- Inherits from `ModelRunner` (defines the strategy interface)
- Implements `run_inference_job()` with Model1-specific logic
- Model1-specific preprocessing (grayscale, 512x512)
- Model1-specific inference (sigmoid + threshold)

---

### Step 3: Create Model2 Runner (Concrete Strategy 2)

**File: `backend/services/runners/model2_runner.py`**

```python
from services.model_runner_base import ModelRunner
from db import get_db, get_fs
from bson.objectid import ObjectId
import datetime
from PIL import Image
import numpy as np
import io
import tifffile
from services.storage import save_bytes_to_gridfs
from flask import current_app
import torch

class Model2Runner(ModelRunner):
    """Concrete Strategy 2: Model2 with different preprocessing and inference logic."""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def load_model(self, model_path):
        """Load Model2 from file."""
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        return self.model
    
    def preprocess_image(self, image_bytes):
        """
        Model2-specific preprocessing.
        
        Example: Model2 expects:
        - RGB or TIFF images
        - 1024x1024 resolution
        - ImageNet normalization
        """
        # Handle both TIFF and regular image formats
        try:
            image_array = tifffile.imread(io.BytesIO(image_bytes))
        except:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_array = np.array(image)
        
        # Resize
        image = Image.fromarray(image_array)
        image = image.resize((1024, 1024))
        image_array = np.array(image, dtype=np.float32)
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array / 255.0 - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def run_model2_inference(self, image_tensor):
        """Model2-specific inference logic."""
        with torch.no_grad():
            output = self.model(image_tensor)
            # Model2 outputs multi-class segmentation
            mask = torch.argmax(output, dim=1)
            mask_array = mask.cpu().numpy()[0]
        
        return mask_array.astype(np.uint8)
    
    def run_inference_job(self, inference_id_str: str):
        """
        Execute Model2 inference.
        This method is called by the dispatcher when user selects Model2.
        """
        inference_id = ObjectId(inference_id_str)
        
        try:
            self.update_inference_status(inference_id, "running")
            current_app.logger.info(f"[Model2] Starting inference: {inference_id_str}")
            
            # Get documents
            inference_doc = self.db.inferences.find_one({"_id": inference_id})
            dataset_doc = self.db.datasets.find_one({"_id": inference_doc['dataset_id']})
            model_doc = self.db.models.find_one({"_id": inference_doc['model_id']})
            
            # Load model
            model_file = self.fs.get(model_doc['artifact_id'])
            model_path = f"/tmp/model2_{inference_id}.pth"
            with open(model_path, 'wb') as f:
                f.write(model_file.read())
            
            self.load_model(model_path)
            
            results = []
            
            # Process each image
            for file_ref in dataset_doc.get('files', []):
                if file_ref['type'] == 'image':
                    try:
                        image_file = self.fs.get(file_ref['gridfs_id'])
                        image_bytes = image_file.read()
                        
                        # Preprocess (Model2-specific)
                        image_tensor = self.preprocess_image(image_bytes)
                        
                        # Run inference (Model2-specific)
                        mask_array = self.run_model2_inference(image_tensor)
                        
                        # Save result
                        mask_img = Image.fromarray(mask_array)
                        mask_bytes_io = io.BytesIO()
                        mask_img.save(mask_bytes_io, format='PNG')
                        mask_bytes = mask_bytes_io.getvalue()
                        
                        mask_filename = f"mask_{file_ref['filename'].rsplit('.', 1)[0]}.png"
                        mask_gridfs_id = save_bytes_to_gridfs(
                            mask_bytes,
                            filename=mask_filename,
                            metadata={
                                'type': 'mask',
                                'model': 'model2',
                                'source_image_gridfs_id': str(file_ref['gridfs_id']),
                                'inference_id': str(inference_id)
                            }
                        )
                        
                        results.append({
                            "source_filename": file_ref['filename'],
                            "source_image_gridfs_id": str(file_ref['gridfs_id']),
                            "class_mask_id": str(mask_gridfs_id),
                            "instance_mask_id": None
                        })
                        
                        current_app.logger.info(f"[Model2] Processed: {file_ref['filename']}")
                    
                    except Exception as e:
                        current_app.logger.error(f"[Model2] Error processing {file_ref['filename']}: {e}")
                        continue
            
            self.update_inference_status(inference_id, "completed", results=results)
            current_app.logger.info(f"[Model2] Inference completed with {len(results)} results")
        
        except Exception as e:
            current_app.logger.error(f"[Model2] Inference failed: {e}")
            self.update_inference_status(inference_id, "failed", error=str(e))
    
    def retrain_job(self, annotation_document_id_str: str, model_id_str: str):
        """Model2-specific retraining logic."""
        current_app.logger.info(f"[Model2] Starting retraining...")
        # Implement Model2 retraining
        pass
```

**Key Differences from Model1**:
- Different preprocessing (RGB/TIFF, 1024x1024, ImageNet normalization)
- Different inference (argmax for multi-class instead of sigmoid)
- Handles both TIFF and regular images

---

### Step 4: Dynamic Dispatcher (The Context)

**File: `backend/services/inference_manager.py`**

```python
from db import get_db
from bson.objectid import ObjectId
from services.runners.cellpose_runner import CellposeRunner
from services.runners.custom_tiff_runner import CustomTiffRunner
from services.runners.model1_runner import Model1Runner
from services.runners.model2_runner import Model2Runner
import datetime
from flask import current_app

# RUNNER REGISTRY - Maps runner_name to runner class
# This is the key to the Strategy Pattern!
RUNNER_REGISTRY = {
    "cellpose": CellposeRunner,
    "custom_tiff": CustomTiffRunner,
    "model1": Model1Runner,    # User uploads model with runner_name="model1"
    "model2": Model2Runner,    # User uploads model with runner_name="model2"
}


def start_managed_inference(inference_id_str: str):
    """
    THIS IS THE DISPATCHER - The Context that selects the strategy at runtime.
    
    How it works:
    1. User selects model in frontend (e.g., "Medical Image Segmenter")
    2. Frontend sends inference request with model_id
    3. This function looks up the model
    4. Extracts runner_name from model metadata
    5. Uses RUNNER_REGISTRY to get the correct runner class
    6. Instantiates and executes the runner
    
    This is DYNAMIC DISPATCH - no hardcoded if/elif statements!
    """
    db = get_db()
    inference_id = ObjectId(inference_id_str)
    
    try:
        current_app.logger.info(f"[DISPATCHER] Starting inference: {inference_id_str}")
        
        # Step 1: Get the inference request
        inference_doc = db.inferences.find_one({"_id": inference_id})
        if not inference_doc:
            raise ValueError("Inference job not found")
        
        current_app.logger.info(f"[DISPATCHER] Found inference document")
        
        # Step 2: Get the model the user selected
        model_doc = db.models.find_one({"_id": inference_doc['model_id']})
        if not model_doc:
            raise ValueError("Model not found")
        
        current_app.logger.info(f"[DISPATCHER] Found model: {model_doc['name']}")
        
        # Step 3: Extract runner_name from model metadata
        runner_name = model_doc.get("runner_name")
        current_app.logger.info(f"[DISPATCHER] Model runner_name: {runner_name}")
        
        # Step 4: Check if runner is registered
        if runner_name not in RUNNER_REGISTRY:
            raise NotImplementedError(
                f"Runner '{runner_name}' is not implemented. "
                f"Available runners: {list(RUNNER_REGISTRY.keys())}"
            )
        
        # Step 5: DYNAMIC DISPATCH - Get the correct runner class from registry
        runner_class = RUNNER_REGISTRY[runner_name]
        current_app.logger.info(f"[DISPATCHER] Selected runner class: {runner_class.__name__}")
        
        # Step 6: Instantiate the runner
        runner = runner_class()
        current_app.logger.info(f"[DISPATCHER] Instantiated {runner_class.__name__}")
        
        # Step 7: Execute the runner's inference method
        current_app.logger.info(f"[DISPATCHER] Calling run_inference_job()")
        runner.run_inference_job(inference_id_str)
        
        current_app.logger.info(f"[DISPATCHER] Inference {inference_id_str} completed successfully")
    
    except Exception as e:
        current_app.logger.error(f"[DISPATCHER] Inference {inference_id_str} failed: {e}")
        db.inferences.update_one(
            {"_id": inference_id},
            {
                "$set": {
                    "status": "failed",
                    "finished_at": datetime.datetime.utcnow(),
                    "notes": str(e)
                }
            }
        )


def start_managed_retraining(annotation_document_id_str: str, model_id_str: str):
    """
    Similar dispatcher for retraining tasks.
    Uses the same RUNNER_REGISTRY to dispatch to correct runner.
    """
    db = get_db()
    model_id = ObjectId(model_id_str)
    
    try:
        current_app.logger.info(f"[DISPATCHER] Starting retraining")
        
        model_doc = db.models.find_one({"_id": model_id})
        if not model_doc:
            raise ValueError("Model not found")
        
        runner_name = model_doc.get("runner_name")
        current_app.logger.info(f"[DISPATCHER] Model runner_name: {runner_name}")
        
        if runner_name not in RUNNER_REGISTRY:
            raise NotImplementedError(f"Runner '{runner_name}' is not implemented")
        
        # DYNAMIC DISPATCH
        runner_class = RUNNER_REGISTRY[runner_name]
        runner = runner_class()
        
        current_app.logger.info(f"[DISPATCHER] Calling retrain_job() on {runner_class.__name__}")
        runner.retrain_job(annotation_document_id_str, model_id_str)
    
    except Exception as e:
        current_app.logger.error(f"[DISPATCHER] Retraining failed: {e}")
        raise
```

**Key Concepts**:
- `RUNNER_REGISTRY`: Maps runner_name → runner class (Factory)
- `start_managed_inference()`: The dispatcher/context that selects the strategy
- No `if/elif/else` statements - uses dynamic lookup instead
- Extensible: Add new runners just by updating the registry

---

### Step 5: Update Main App

**File: `backend/app.py`**

```python
from flask import Flask, jsonify
from config import config
from db import init_db
import commands
from blueprints.auth import auth_bp, create_auth_indexes
from blueprints.datasets import datasets_bp
from blueprints.models_bp import models_bp
from blueprints.inferences import inferences_bp
from blueprints.files import files_bp
from blueprints.cvat_bp import cvat_bp
from blueprints.retraining_bp import retraining_bp

def create_app(config_name='default'):
    """Flask application factory."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    init_db(app)
    commands.register_commands(app)
    
    # Create auth indexes
    from blueprints.auth import create_auth_indexes
    create_auth_indexes(app)
    
    # Register all blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(datasets_bp, url_prefix='/api/datasets')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(inferences_bp, url_prefix='/api/inferences')
    app.register_blueprint(files_bp, url_prefix='/api/files')
    app.register_blueprint(cvat_bp, url_prefix='/api/cvat')
    app.register_blueprint(retraining_bp, url_prefix='/api/retraining')
    
    @app.route('/health')
    def health_check():
        return jsonify({"status": "ok"})
    
    return app
```

---

### Step 6: Update Inference Blueprint

**File: `backend/blueprints/inferences.py`**

```python
from flask import Blueprint, request, jsonify, current_app
from bson.objectid import ObjectId
from db import get_db, get_fs
from utils.security import jwt_required
import datetime
import threading
from services.inference_manager import start_managed_inference

inferences_bp = Blueprint('inferences', __name__)


@inferences_bp.route('/start', methods=['POST'])
@jwt_required
def start_inference(current_user_id):
    """
    Start inference with automatically selected model runner.
    
    Request:
    {
        "dataset_id": "string",
        "model_id": "string",  # User selects which model to use
        "params": {}
    }
    """
    db = get_db()
    data = request.json
    
    dataset_id = data.get('dataset_id')
    model_id = data.get('model_id')
    params = data.get('params', {})
    
    if not dataset_id or not model_id:
        return jsonify({'error': 'dataset_id and model_id required'}), 400
    
    try:
        dataset_id_obj = ObjectId(dataset_id)
        model_id_obj = ObjectId(model_id)
    except Exception:
        return jsonify({'error': 'Invalid ObjectId format'}), 400
    
    try:
        # Verify dataset exists
        dataset = db.datasets.find_one({'_id': dataset_id_obj})
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Verify model exists
        model = db.models.find_one({'_id': model_id_obj})
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Create inference document
        inference_doc = {
            'dataset_id': dataset_id_obj,
            'model_id': model_id_obj,
            'requested_by': ObjectId(current_user_id),
            'params': params,
            'status': 'queued',
            'created_at': datetime.datetime.utcnow()
        }
        
        result = db.inferences.insert_one(inference_doc)
        inference_id = result.inserted_id
        
        # Start inference in background thread
        # The dispatcher will automatically route to correct runner
        thread = threading.Thread(
            target=start_managed_inference,
            args=(str(inference_id),)
        )
        thread.daemon = False
        thread.start()
        
        return jsonify({
            'success': True,
            'inference_id': str(inference_id),
            'status': 'queued',
            'message': f'Inference started using {model["name"]}'
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error starting inference: {e}")
        return jsonify({'error': f'Failed to start inference: {str(e)}'}), 500


@inferences_bp.route('/<inference_id>', methods=['GET'])
@jwt_required
def get_inference(current_user_id, inference_id):
    """Get inference results."""
    db = get_db()
    
    try:
        inference = db.inferences.find_one({'_id': ObjectId(inference_id)})
        if not inference:
            return jsonify({'error': 'Inference not found'}), 404
        
        # Check authorization
        if str(inference['requested_by']) != current_user_id:
            return jsonify({'error': 'Forbidden'}), 403
        
        # Convert ObjectIds to strings
        inference['_id'] = str(inference['_id'])
        inference['dataset_id'] = str(inference['dataset_id'])
        inference['model_id'] = str(inference['model_id'])
        inference['requested_by'] = str(inference['requested_by'])
        
        return jsonify(inference), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get inference: {str(e)}'}), 500
```

---

## Complete Workflow: Strategy Pattern in Action

### Step-by-Step Execution Flow

**1. User Selects Model in Frontend**
```javascript
// User sees dropdown with models
// [✓] Medical Image Segmenter (Model1)
// [ ] Microscopy Cell Segmenter (Model2)
// User clicks "Run Inference" with Model1 selected
```

**2. Frontend Sends Request**
```bash
POST /api/inferences/start
{
  "dataset_id": "abc123",
  "model_id": "xyz789",  # Model1's ID
  "params": {}
}
```

**3. Backend Creates Inference Record**
```python
# app.py receives request
inference_doc = {
  'dataset_id': ObjectId('abc123'),
  'model_id': ObjectId('xyz789'),  # Model1
  'requested_by': ObjectId(user_id),
  'status': 'queued',
  'created_at': datetime.now()
}
```

**4. Dispatcher is Called**
```python
# inferences.py calls start_managed_inference()
start_managed_inference('inference_id')
```

**5. Dispatcher Looks Up Model**
```python
# inference_manager.py
inference_doc = db.inferences.find_one({'_id': inference_id})
model_doc = db.models.find_one({'_id': inference_doc['model_id']})

# Result: model_doc = {
#   'name': 'Medical Image Segmenter',
#   'runner_name': 'model1',  # KEY!
#   ...
# }
```

**6. Dynamic Dispatch (Strategy Selection)**
```python
runner_name = model_doc.get('runner_name')  # 'model1'
runner_class = RUNNER_REGISTRY[runner_name]  # Model1Runner class
runner = runner_class()  # Instantiate Model1Runner
```

**7. Execute Model1-Specific Logic**
```python
runner.run_inference_job(inference_id_str)

# Calls Model1Runner.run_inference_job()
# Which does:
#   - Model1.preprocess_image() [grayscale, 512x512]
#   - Model1.run_model1_inference() [sigmoid + threshold]
#   - Saves results
```

**8. Update Status**
```python
db.inferences.update_one(
    {'_id': inference_id},
    {'$set': {'status': 'completed', 'results': [...]}
)
```

**9. Frontend Polls for Results**
```javascript
// Frontend periodically calls:
GET /api/inferences/{inference_id}
// Gets: { status: 'completed', results: [...] }
```

---

## Key Benefits of Strategy Pattern

| Benefit | Description |
|---------|-------------|
| **Extensibility** | Add Model3, Model4 by just creating runner + updating registry |
| **Flexibility** | Switch models without changing frontend |
| **Maintainability** | Each model's code isolated in its own file |
| **Testability** | Each runner can be tested independently |
| **Clean Code** | No massive if/elif/else chains |
| **Runtime Selection** | Model chosen at runtime based on user input |

---

## File Structure

```
backend/
├── services/
│   ├── inference_manager.py         # Dispatcher (CORE FILE)
│   ├── model_runner_base.py         # Base class
│   └── runners/
│       ├── model1_runner.py         # Concrete strategy 1
│       ├── model2_runner.py         # Concrete strategy 2
│       ├── cellpose_runner.py       # Concrete strategy 3
│       └── ...
└── blueprints/
    └── inferences.py                # Calls dispatcher
```

---

## How to Add a New Model

**To add Model3:**

1. **Create runner**: `backend/services/runners/model3_runner.py`
```python
class Model3Runner(ModelRunner):
    def run_inference_job(self, inference_id_str):
        # Model3 logic
        pass
```

2. **Register**: Update `backend/services/inference_manager.py`
```python
from services.runners.model3_runner import Model3Runner

RUNNER_REGISTRY = {
    ...
    "model3": Model3Runner,  # ADD THIS
}
```

3. **Upload Model**: 
```bash
POST /api/models/upload
  runner_name: "model3"  # Matches registry key
```

**That's it!** Frontend doesn't need changes, backend automatically routes!

---

## Important: Upload Model with runner_name

When uploading models to your app, you MUST set the `runner_name` field:

```bash
# Model1
POST /api/models/upload
  name: "Medical Image Segmenter"
  runner_name: "model1"  # Must match RUNNER_REGISTRY key
  
# Model2
POST /api/models/upload
  name: "Microscopy Cell Segmenter"
  runner_name: "model2"  # Must match RUNNER_REGISTRY key
```

The `runner_name` is what tells the dispatcher which runner to use!
