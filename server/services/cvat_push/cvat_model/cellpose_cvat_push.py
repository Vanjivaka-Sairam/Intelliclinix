
from ..cvat_push_base import CvatBase
import io
import zipfile
from bson import ObjectId
from typing import List, Dict

class CellposeModel(CvatBase):
    
    def get_labels(self) -> List[Dict[str, str]]:
        return [
            {"name": "nucleus", "color": "#8A119D"},
            {"name": "cell", "color": "#00FF00"},
            {"name": "background", "color": "#000000"}
        ]
    
    def load_data(self, run_id: str):
        run_doc = self.db.inferences.find_one({"_id": ObjectId(run_id)})
        if not run_doc:
            raise ValueError(f"Run document with ID {run_id} not found")
        
        image_files = []
        image_data_map = {}
        
        for result in run_doc.get("results", []):
            source_filename = result.get("source_filename")
            source_image_gridfs_id = result.get("source_image_gridfs_id")
            
            if source_filename and source_image_gridfs_id:
                grid_out = self.fs.get(ObjectId(source_image_gridfs_id))
                image_data = grid_out.read()
                
                file_obj = io.BytesIO(image_data)
                file_obj.name = source_filename
                image_files.append(file_obj)
                
                image_data_map[source_filename] = {
                    "file_obj": file_obj,
                    "class_mask_id": result.get("class_mask_id"),
                    "instance_mask_id": result.get("instance_mask_id"),
                    "artifacts": result.get("artifacts", [])
                }
        
        return image_files, image_data_map
    
    def prepare_annotations(self, image_data_map: Dict[str, Dict]) -> io.BytesIO:
        import zipfile
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            ids_list = []
            
            for filename, data in image_data_map.items():
                image_id = filename.split('.')[0]
                ids_list.append(image_id)
                
                # Load class mask
                if data.get("class_mask_id"):
                    class_mask = self.fs.get(ObjectId(data["class_mask_id"]))
                    zf.writestr(f"SegmentationClass/{image_id}.png", class_mask.read())
                
                # Load instance mask
                if data.get("instance_mask_id"):
                    instance_mask = self.fs.get(ObjectId(data["instance_mask_id"]))
                    zf.writestr(f"SegmentationObject/{image_id}.png", instance_mask.read())
            
            zf.writestr("ImageSets/Segmentation/default.txt", "\n".join(ids_list))
            
            # Cellpose-specific label map
            labelmap_content = (
                "# label:color_rgb:parts:actions\n"
                "background:0,0,0::\n"
                "nucleus:138,17,157::\n"
                "cell:0,255,0::\n"
            )
            zf.writestr("labelmap.txt", labelmap_content)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def get_annotation_format(self) -> str:
        return "PASCAL VOC 1.1"