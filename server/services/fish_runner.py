import os
import sys
import zipfile
import shutil
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math



#----------------------------configs--------------------------#
# Constants for the pipeline
CELL_CLASS_ID = 4
FUSION_CLASS_ID = 3
GENE_MAP = {0: "green", 1: "red", 2: "aqua"}

# Detection Thresholds
CELL_CONF = 0.5
GENE_CONF = 0.5
#------------------------fish--utils----------------------------
#image processing
def yolo_to_cv2(yolo_box, img_shape):
    """Converts YOLO (cx, cy, w, h) normalized to CV2 (x1, y1, x2, y2) pixels."""
    h, w = img_shape[:2]
    cx_n, cy_n, w_n, h_n = yolo_box
    x1 = int((cx_n - w_n/2) * w)
    y1 = int((cy_n - h_n/2) * h)
    x2 = int((cx_n + w_n/2) * w)
    y2 = int((cy_n + h_n/2) * h)
    return (x1, y1, x2, y2)

def cv2_to_yolo(cv2_box, img_shape, class_id):
    """Converts CV2 pixels to YOLO normalized format with a specific class ID."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = cv2_box
    bw, bh = x2 - x1, y2 - y1
    # Ensure values stay within 0-1 range
    return (
        int(class_id), 
        max(0, min(1, (x1 + bw/2)/w)), 
        max(0, min(1, (y1 + bh/2)/h)), 
        max(0, min(1, bw/w)), 
        max(0, min(1, bh/h))
    )

def translate_to_global(local_cv2_box, cell_global_cv2_box):
    """
    Translates coordinates from a local cell patch back to the global image.
    
    Args:
        local_cv2_box: (x1, y1, x2, y2) inside the patch.
        cell_global_cv2_box: (x1, y1, x2, y2) of the cell in the global image.
    """
    gx1, gy1, _, _ = cell_global_cv2_box
    lx1, ly1, lx2, ly2 = local_cv2_box
    
    return (lx1 + gx1, ly1 + gy1, lx2 + gx1, ly2 + gy1)
# --------------------------------main_code---------------------------------------------------------------

#fusion

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def get_diag(box):
    x1, y1, x2, y2 = box
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def get_distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def apply_fusion_logic(detections, img_shape):
    """
    Processes local detections to find fusions and remaps IDs.
    IDs: 0: Green, 1: Red, 2: Aqua, 3: Fusion.
    """
    greens = [d for d in detections if d['cid'] == 0]
    reds = [d for d in detections if d['cid'] == 1]
    aquas = [d for d in detections if d['cid'] == 2]

    used_g, used_r = set(), set()
    final_results = []
    
    # 1. Match Green and Red for Fusion (ID 3)
    pairs = []
    for i, g in enumerate(greens):
        for j, r in enumerate(reds):
            dist = get_distance(get_center(g['box']), get_center(r['box']))
            thresh = (get_diag(g['box']) + get_diag(r['box'])) / 2.0
            if dist <= thresh:
                pairs.append((dist, i, j))
    
    pairs.sort() # Prioritize closest pairs
    for _, gi, rj in pairs:
        if gi in used_g or rj in used_r: continue
        used_g.add(gi); used_r.add(rj)
        
        gc, rc = get_center(greens[gi]['box']), get_center(reds[rj]['box'])
        fc = ((gc[0]+rc[0])/2, (gc[1]+rc[1])/2)
        # Scale for fusion box size
        side = (get_diag(greens[gi]['box']) + get_diag(reds[rj]['box'])) / 2.828
        f_box = (int(fc[0]-side), int(fc[1]-side), int(fc[0]+side), int(fc[1]+side))
        
        final_results.append({'cid': 3, 'box': f_box})

    # 2. Add remaining individual signals
    for i, g in enumerate(greens):
        if i not in used_g: final_results.append(g)
    for i, r in enumerate(reds):
        if i not in used_r: final_results.append(r)
    for a in aquas:
        final_results.append(a)

    return final_results


#------------------------------------------------------------------------------------------------------------------------

from services.model_runner_base import ModelRunner
from services.storage import save_bytes_to_gridfs
from flask import current_app
import io
from PIL import Image
from bson.objectid import ObjectId


class FishRunner(ModelRunner):
    def run_inference_job(self, inference_id_str) -> None:
        inference_id = ObjectId(inference_ud_str)

        self.db.inferences.update_one({
            {"_id" : inference_id},
            {"$set" : {"status" : "running"}}
        })

        #load model paths
        


