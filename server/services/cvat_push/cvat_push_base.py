from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from pymongo import MongoClient
import gridfs
from bson import ObjectId
import io

class CvatBase(ABC): #base model for other models
    
    def __init__(self, db, fs):
        self.db = db
        self.fs = fs
    
    @abstractmethod
    def get_labels(self) -> List[Dict[str, str]]:
        """Return model-specific labels with colors"""
        pass
    
    @abstractmethod
    def load_data(self, run_id: str) -> Tuple[List[io.BytesIO], Dict[str, Dict]]:
        """Load images and masks from MongoDB for this model"""
        pass
    
    @abstractmethod
    def prepare_annotations(self, image_data_map: Dict[str, Dict]) -> io.BytesIO:
        """Prepare annotations in model-specific format"""
        pass
    
    @abstractmethod
    def get_annotation_format(self) -> str:
        """Return CVAT annotation format (PASCAL VOC, COCO, etc.)"""
        pass