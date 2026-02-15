# ---------------------------------------------------------------
# Inference → Overlay like GT figure + class-colored HOLLOW RINGS
# ---------------------------------------------------------------
# - Cellpose nuclei (RAW patch) → instance mask
# - ROIAlign RAW crops → YOLOv8 → Map back
# - Draw hollow class-colored circles
# - Side-by-side figure + summary table + CSV
# ===============================================================

import os, csv
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from cellpose.models import CellposeModel
from ultralytics import YOLO


from services.model_runner_base import ModelRunner
from services.storage import save_bytes_to_gridfs
from flask import current_app
import io
from PIL import Image
from bson.objectid import ObjectId

# -------------------- USER CONFIG --------------------
IMAGES_DIR = "./images"
OUT_DIR    = ""

# Load defaults from env or use relative fallback (for standalone use)
CELLPOSE_WEIGHTS = os.getenv("DDISH_CELLPOSE_MODEL_PATH", "./D-DISH_nuclei_v1")
YOLO_WEIGHTS     = os.getenv("DDISH_YOLO_MODEL_PATH", "./Yolo_val6291SJ.pt")

PATCH_SIZE   = (384, 384)   # ROIAlign (H, W)
YOLO_IMGSZ   = 384  
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
MIN_BBOX_SZ_NUC   = 8          
YOLO_CONF         = 0.30
YOLO_IOU          = 0.45
CENTER_DIST_MIN   = 4.0        
MAX_AR            = 6.0        
MAX_AREA_FRAC     = 0.40       

# -------------------- Label & color config --------------------
# Map class IDs to names (must match your YOLO training label order!!!)
CLASS_NAMES: Dict[int, str] = {
    0: "HER2_Cluster3",
    1: "HER2_Cluster6",
    2: "HER2_Cluster12",
    3: "Fusion(Her2+Chr17)",
    4: "Chr17",
    5: "Her2",
}

# Marker palette (RGB) -> converted to BGR for OpenCV; index by class id
PALETTE: List[Tuple[int,int,int]] = [
    (184,61,245), (89,89,255), (0,255,230), (250,250,55),
    (240,147,167), (0,0,0)
]

# Per-nucleus overlay colors (RGB)
NUCLEUS_PALETTE: List[Tuple[int,int,int]] = [
    (31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),
    (140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207),
    (174,199,232),(255,187,120),(152,223,138),(255,152,150),(197,176,213),
    (196,156,148),(247,182,210),(199,199,199),(219,219,141),(158,218,229)
]

# Aesthetics
FIG_DPI = 300
FONT_SIZE_TITLE = 12
FONT_SIZE_TABLE = 10

# Nucleus appearance
NUCLEUS_EDGE_THICKNESS = 1          # thin boundary around the nuclues
NUCLEUS_FILL_ALPHA = 0.25           # "a little" opacity inside each nucleus

# Markers 
MARKER_RADIUS = 4
MARKER_THICKNESS = 1

PER_NUCLEUS_ALPHA  = NUCLEUS_FILL_ALPHA
PER_NUCLEUS_BORDER = NUCLEUS_EDGE_THICKNESS

# Globals for legend (populated in main)
PRETTY_ORDER: List[str] = []
PRETTY2COLOR_BGR: Dict[str, Tuple[int,int,int]] = {}

# -------------------- helpers --------------------(do not edit/mess with the helper function: Aman)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

@torch.no_grad()
def cellpose_segment(model: CellposeModel, rgb: np.ndarray) -> np.ndarray:
    masks, flows, styles = model.eval(
        rgb.astype(np.float32), channels=[0, 0], diameter=None,
        flow_threshold=0.4, cellprob_threshold=0.0
    )
    return masks.astype(np.int32)

def masks_to_bboxes(inst_mask: np.ndarray, min_sz=MIN_BBOX_SZ_NUC) -> List[Tuple[int,int,int,int,int]]:
    ids = np.unique(inst_mask); ids = ids[ids != 0]
    out = []
    for nid in ids:
        ys, xs = np.where(inst_mask == nid)
        if xs.size == 0: continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
        if (x2-x1) < min_sz or (y2-y1) < min_sz: continue
        out.append((int(nid), int(x1), int(y1), int(x2), int(y2)))
    return out

def _rgb_to_bgr(c_rgb: Tuple[int,int,int]) -> Tuple[int,int,int]:
    return (c_rgb[2], c_rgb[1], c_rgb[0])

def _nucleus_color_bgr(nid: int) -> Tuple[int,int,int]:
    rgb = NUCLEUS_PALETTE[(nid - 1) % len(NUCLEUS_PALETTE)]
    return _rgb_to_bgr(rgb)

def apply_overlay_multicolor(rgb: np.ndarray, inst_mask: np.ndarray,
                             alpha: float, border_px: int) -> np.ndarray:
    """Per-nucleus soft fill + thin (configurable) boundary."""
    base_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    ids = np.unique(inst_mask); ids = ids[ids != 0]
    for nid in ids:
        m = (inst_mask == nid).astype(np.uint8) * 255
        color_bgr = _nucleus_color_bgr(nid)

        if alpha > 0:
            fill = np.zeros_like(base_bgr, dtype=np.uint8); fill[:] = color_bgr
            masked_fill = cv2.bitwise_and(fill, fill, mask=m)
            base_bgr = cv2.addWeighted(base_bgr, 1.0, masked_fill, alpha, 0)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base_bgr, cnts, -1, color_bgr,
                         thickness=border_px, lineType=cv2.LINE_AA)

    return cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)

def roi_align_crops(src_rgb: np.ndarray, boxes_xyxy: List[Tuple[int,int,int,int]],
                    out_h: int, out_w: int, device: str):
    H, W, _ = src_rgb.shape
    img_t = torch.from_numpy(src_rgb).to(device=device, dtype=torch.float32) / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # 1×3×H×W
    rois, meta = [], []
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W-1, x2), min(H-1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        rois.append([0, float(x1c), float(y1c), float(x2c), float(y2c)])
        roi_w, roi_h = (x2c - x1c), (y2c - y1c)
        sx = roi_w / float(out_w); sy = roi_h / float(out_h)
        meta.append((x1c, y1c, sx, sy, roi_w, roi_h))
    if not rois:
        return torch.empty(0,3,out_h,out_w,device=device), meta
    rois_t = torch.tensor(rois, dtype=torch.float32, device=device)
    crops = roi_align(img_t, rois_t, output_size=(out_h, out_w),
                      spatial_scale=1.0, sampling_ratio=-1, aligned=True)
    return crops, meta

def map_patch_dets_to_full(dets_xyxy_patch: np.ndarray, roi_meta):
    x1, y1, sx, sy, _, _ = roi_meta
    out = dets_xyxy_patch.copy()
    out[:, [0,2]] = x1 + out[:, [0,2]] * sx
    out[:, [1,3]] = y1 + out[:, [1,3]] * sy
    return out

def build_dt_maps_and_areas(inst_mask: np.ndarray, id_boxes):
    dt_maps, areas = {}, {}  
    for nid, x1, y1, x2, y2 in id_boxes:
        local = (inst_mask[y1:y2, x1:x2] == nid).astype(np.uint8) * 255
        if local.size == 0: continue
        dt = cv2.distanceTransform(local, cv2.DIST_L2, 3)
        dt_maps[nid] = (dt, x1, y1)
        areas[nid] = int((local > 0).sum())
    return dt_maps, areas

def center_ok(nid: int, cx: int, cy: int, dt_maps, min_dist: float) -> bool:
    if nid not in dt_maps: return False
    dt, x1, y1 = dt_maps[nid]; lx, ly = cx - x1, cy - y1
    if lx < 0 or ly < 0 or ly >= dt.shape[0] or lx >= dt.shape[1]: return False
    return dt[ly, lx] >= min_dist

def shape_filters(nid: int, xyxy, areas, dt_maps) -> Tuple[bool, List[float]]:
    x1, y1, x2, y2 = map(float, xyxy)
    w, h = x2-x1, y2-y1
    if w <= 0 or h <= 0: return False, [x1, y1, x2, y2]
    cx, cy = int((x1+x2)/2.0), int((y1+y2)/2.0)
    if not center_ok(nid, cx, cy, dt_maps, CENTER_DIST_MIN): 
        return False, [x1, y1, x2, y2]
    ar = max(w/h, h/w)
    if ar > MAX_AR: return False, [x1, y1, x2, y2]
    if nid in areas and areas[nid] > 0:
        area_frac = (w*h)/float(areas[nid])
        if area_frac > MAX_AREA_FRAC: return False, [x1, y1, x2, y2]
    return True, [x1, y1, x2, y2]

def build_class_maps_from_user_config() -> Tuple[Dict[int,str], Dict[int,Tuple[int,int,int]]]:
    id2pretty = dict(CLASS_NAMES)

    id2color: Dict[int, Tuple[int,int,int]] = {}
    for i, _name in id2pretty.items():
        if i < len(PALETTE):
            id2color[i] = _rgb_to_bgr(PALETTE[i])
        else:
            id2color[i] = (0, 255, 0)  # fallback
    return id2pretty, id2color

def draw_marker_hollow_circle(img_bgr: np.ndarray, xyxy, color_bgr):
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
    cv2.circle(img_bgr, (cx, cy), int(MARKER_RADIUS), color_bgr,
               thickness=int(MARKER_THICKNESS), lineType=cv2.LINE_AA)

def _rgb_tuple_from_bgr(bgr): return (bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0)

def make_compare_figure(img_name: str, orig_rgb: np.ndarray, overlay_bgr_draw: np.ndarray,
                        nuclei_count: int, counts_by_pretty: Dict[str,int], save_path: Path):
    ov_rgb = cv2.cvtColor(overlay_bgr_draw, cv2.COLOR_BGR2RGB)
    H, W, _ = orig_rgb.shape
    fig = plt.figure(figsize=(12, 6.5), dpi=FIG_DPI)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.9], height_ratios=[0.12, 0.88], figure=fig)

    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis('off')
    ax_title.text(0.5, 0.5, f"{img_name}   |   {W}×{H}px",
                  ha='center', va='center', fontsize=FONT_SIZE_TITLE, fontweight='bold')

    ax1 = fig.add_subplot(gs[1, 0]); ax1.imshow(orig_rgb); ax1.axis('off'); ax1.set_title("Original", fontsize=FONT_SIZE_TABLE, pad=6)
    ax2 = fig.add_subplot(gs[1, 1]); ax2.imshow(ov_rgb);  ax2.axis('off'); ax2.set_title("Per-nucleus overlay + Markers", fontsize=FONT_SIZE_TABLE, pad=6)

    ax3 = fig.add_subplot(gs[1, 2]); ax3.axis('off')
    total_markers = sum(counts_by_pretty.values())
    table_data = [
        ["Nuclei (mask)", nuclei_count],
        ["Total markers", total_markers],
    ]
    for label in PRETTY_ORDER:
        table_data.append([label, counts_by_pretty.get(label, 0)])

    tbl = ax3.table(cellText=table_data, colLabels=["Object", "Count"],
                    loc="center", cellLoc='left', colLoc='left')
    tbl.scale(1.05, 1.25)
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if key[0] == 0:
            cell.set_facecolor('#f2f2f2'); cell.set_text_props(weight='bold'); cell.set_fontsize(FONT_SIZE_TABLE)
        else:
            cell.set_fontsize(FONT_SIZE_TABLE)

    legend_handles = [Patch(facecolor='none', edgecolor=_rgb_tuple_from_bgr(PRETTY2COLOR_BGR[p]), label=p)
                      for p in PRETTY_ORDER]
    ax3.legend(handles=legend_handles, title="Class colors", loc='lower center', bbox_to_anchor=(0.5, -0.05),
               frameon=True, fontsize=FONT_SIZE_TABLE, title_fontsize=FONT_SIZE_TABLE)

    ensure_dir(save_path.parent); fig.savefig(str(save_path), bbox_inches='tight'); plt.close(fig)



def generate_interactive_data(inst_mask: np.ndarray, detections_arg: List[dict], image_id: str) -> dict:
    """
    Generates a JSON structure for the interactive frontend viewer.
    :param inst_mask: 2D integer array (Cellpose output)
    :param detections_arg: List of dicts representing ALL detections for this image (from all_rows)
    :param image_id: String identifier (filename or ID)
    """
    output_data = {"image_id": image_id, "nuclei": []}
    
    # Get all unique Nucleus IDs (skip 0=background)
    ids = np.unique(inst_mask)
    ids = ids[ids != 0]
    
    for nid in ids:
        # Create binary mask for current nucleus
        # Optimization: Isolate ROI to speed up findContours
        rows, cols = np.where(inst_mask == nid)
        if len(rows) == 0: continue
        
        y1, x1 = np.min(rows), np.min(cols)
        y2, x2 = np.max(rows), np.max(cols)
        
        # Pad ROI to prevent contour clipping
        roi = inst_mask[y1:y2+1, x1:x2+1]
        binary_roi = (roi == nid).astype(np.uint8)
        
        # Find contours in ROI coordinates
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Simplify contour
            cnt = contours[0]
            epsilon = 0.01 * cv2.arcLength(cnt, True) # 1% error allowed
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Offset contours back to global image coordinates
            # approx shape is (N, 1, 2) -> we want (N, 2)
            global_contour = approx.reshape(-1, 2) + [x1, y1]
            
            # Filter detections for this nucleus
            # detections_arg is a list of dicts. We filter by 'nucleus_id'
            n_dets = [d for d in detections_arg if d['nucleus_id'] == nid]
            
            # Count stats
            her2_count = sum(1 for d in n_dets if d['cls_name_model'] == 'Her2')
            chr17_count = sum(1 for d in n_dets if d['cls_name_model'] == 'Chr17')
            fusion_count = sum(1 for d in n_dets if 'Fusion' in d['cls_name_model'])
            
            nucleus_obj = {
                "id": int(nid),
                "polygon": global_contour.tolist(), # [[x,y], [x,y]...]
                "stats": {
                    "Her2": her2_count,
                    "Chr17": chr17_count,
                    "Fusion": fusion_count
                },
                # We can optionally include detailed markers if needed for the tooltip
                # "markers": [...] 
            }
            output_data["nuclei"].append(nucleus_obj)
            
    return output_data


# -------------------- MAIN RUNNER CLASS --------------------



class DDishRunner(ModelRunner):

    def run_inference_job(self, inference_id_str) -> None:
        inference_id = ObjectId(inference_id_str)

        self.db.inferences.update_one(
            {"_id" : inference_id},
            {"$set" : {"status" : "running"}}
            )
        inference_doc = self.db.inferences.find_one({"_id": inference_id})
        dataset_doc = self.db.datasets.find_one({"_id": inference_doc["dataset_id"]})

        # Load models
        # Load models from configuration
        cellpose_path_str = current_app.config.get("DDISH_CELLPOSE_MODEL_PATH")
        yolo_path_str = current_app.config.get("DDISH_YOLO_MODEL_PATH")

        if not cellpose_path_str or not yolo_path_str:
             raise ValueError("DDISH model paths not configured in .env or config.py")

        cellpose_weights = Path(cellpose_path_str)
        yolo_weights = Path(yolo_path_str)

        if not cellpose_weights.exists():
             raise FileNotFoundError(f"Cellpose weights not found at {cellpose_weights}")
        if not yolo_weights.exists():
             raise FileNotFoundError(f"YOLO weights not found at {yolo_weights}")

        device = DEVICE
        print(f"Loading Cellpose: {cellpose_weights}")
        cp_model = CellposeModel(gpu=(device.startswith("cuda")), pretrained_model=str(cellpose_weights))

        print(f"Loading YOLOv8: {yolo_weights}")
        yolo = YOLO(str(yolo_weights))

        ID2PRETTY, ID2COLOR = build_class_maps_from_user_config()
        
        # Legend order & color map derived from CLASS_NAMES/PALETTE
        global PRETTY_ORDER, PRETTY2COLOR_BGR
        PRETTY_ORDER = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
        PRETTY2COLOR_BGR = {ID2PRETTY[i]: ID2COLOR[i] for i in ID2PRETTY}

        all_rows = []
        results = []

        # Iterate over files in the dataset
        for file_ref in dataset_doc.get("files", []):
             if file_ref.get("type", "") != "image":
                 continue
             
             # Load image from GridFS
             gridfs_id = file_ref["gridfs_id"]
             file_obj = self.fs.get(gridfs_id)
             file_bytes = file_obj.read()
             
             # Convert bytes to numpy array (RGB)
             pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
             rgb = np.array(pil_img)
             img_name = file_ref["filename"]
             img_stem = Path(img_name).stem
             
             # Keep track of detections for this specific image to build the JSON
             current_image_detections = []

             # --- Inference Pipeline ---
             inst_mask = cellpose_segment(cp_model, rgb)

             overlay_rgb = apply_overlay_multicolor(rgb, inst_mask, PER_NUCLEUS_ALPHA, PER_NUCLEUS_BORDER)
             overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR).copy()

             id_boxes = masks_to_bboxes(inst_mask)
             nuclei_count = len(id_boxes)
             dt_maps, nuc_areas = build_dt_maps_and_areas(inst_mask, id_boxes)

             counts_by_pretty: Dict[str,int] = {}

             if nuclei_count > 0:
                xyxys = [(x1,y1,x2,y2) for (_,x1,y1,x2,y2) in id_boxes]
                crops_t, metas = roi_align_crops(rgb, xyxys, out_h=PATCH_SIZE[0], out_w=PATCH_SIZE[1], device=device)
                
                if crops_t.numel() > 0:
                    crops_np_rgb = (crops_t.clamp(0,1).mul(255).byte().permute(0,2,3,1).cpu().numpy())
                    yolo_results = yolo.predict(source=list(crops_np_rgb),
                                           conf=YOLO_CONF, iou=YOLO_IOU,
                                           verbose=False, imgsz=YOLO_IMGSZ)

                    for i, res in enumerate(yolo_results):
                        nid, x1, y1, x2, y2 = id_boxes[i]
                        roi_meta = metas[i]
                        if res is None or res.boxes is None or len(res.boxes) == 0:
                            continue

                        dets_xyxy = res.boxes.xyxy.cpu().numpy()
                        dets_cls  = res.boxes.cls.cpu().numpy().astype(int)
                        dets_conf = res.boxes.conf.cpu().numpy()

                        dets_full = map_patch_dets_to_full(dets_xyxy, roi_meta)

                        for j in range(dets_full.shape[0]):
                            cls_id = int(dets_cls[j])
                            pretty = ID2PRETTY.get(cls_id, f"cls{cls_id}")
                            color  = ID2COLOR.get(cls_id, (0,255,0))

                            keep, box_adj = shape_filters(nid, dets_full[j], nuc_areas, dt_maps)
                            if not keep:
                                continue

                            draw_marker_hollow_circle(overlay_bgr, box_adj, color_bgr=color)
                            counts_by_pretty[pretty] = counts_by_pretty.get(pretty, 0) + 1
                            
                            row_data = {
                                "image": img_name,
                                "nucleus_id": nid,
                                "cls_id": cls_id,
                                "cls_name_model": ID2PRETTY.get(cls_id, f"cls{cls_id}"),
                                "conf": f"{float(dets_conf[j]):.4f}",
                                "x1": f"{box_adj[0]:.1f}", "y1": f"{box_adj[1]:.1f}",
                                "x2": f"{box_adj[2]:.1f}", "y2": f"{box_adj[3]:.1f}",
                            }
                            all_rows.append(row_data)
                            current_image_detections.append(row_data)

             # --- Artifact Generation ---
             
             # 0. Interactive Nuclei JSON (New)
             import json
             nuclei_data = generate_interactive_data(inst_mask, current_image_detections, img_name)
             nuclei_json_bytes = json.dumps(nuclei_data).encode('utf-8')
             nuclei_json_id = save_bytes_to_gridfs(
                 nuclei_json_bytes,
                 filename=f"{img_stem}_nuclei.json",
                 metadata={"inference_id": str(inference_id), "source_image_id": str(gridfs_id), "type": "nuclei_json"}
             )

             # 1. Stitched Overlay
             overlay_enc = cv2.imencode(".png", overlay_bgr)[1].tobytes()
             overlay_id = save_bytes_to_gridfs(
                 overlay_enc, 
                 filename=f"{img_stem}_stitched.png",
                 metadata={"inference_id": str(inference_id), "source_image_id": str(gridfs_id), "type": "overlay_stitched"}
             )

             # 2. Comparison Figure
             # We need to adapt make_compare_figure to return bytes or save to a buffer
             # To avoid modifying the helper too much, we can monkey-patch or adjust it. 
             # Actually, let's just inline the figure logic slightly or use a temporary buffer.
             # Better: adjust make_compare_figure to take a file-like object or return bytes?
             # For now, let's use a BytesIO buffer.
             
             fig_buf = io.BytesIO()
             
             # Re-using logic from make_compare_figure but saving to buffer
             ov_rgb_fig = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
             H, W, _ = rgb.shape
             fig = plt.figure(figsize=(12, 6.5), dpi=FIG_DPI)
             gs = GridSpec(2, 3, width_ratios=[1, 1, 0.9], height_ratios=[0.12, 0.88], figure=fig)

             ax_title = fig.add_subplot(gs[0, :]); ax_title.axis('off')
             ax_title.text(0.5, 0.5, f"{img_stem}   |   {W}×{H}px",
                           ha='center', va='center', fontsize=FONT_SIZE_TITLE, fontweight='bold')

             ax1 = fig.add_subplot(gs[1, 0]); ax1.imshow(rgb); ax1.axis('off'); ax1.set_title("Original", fontsize=FONT_SIZE_TABLE, pad=6)
             ax2 = fig.add_subplot(gs[1, 1]); ax2.imshow(ov_rgb_fig);  ax2.axis('off'); ax2.set_title("Per-nucleus overlay + Markers", fontsize=FONT_SIZE_TABLE, pad=6)

             ax3 = fig.add_subplot(gs[1, 2]); ax3.axis('off')
             total_markers = sum(counts_by_pretty.values())
             table_data = [
                 ["Nuclei (mask)", nuclei_count],
                 ["Total markers", total_markers],
             ]
             for label in PRETTY_ORDER:
                 table_data.append([label, counts_by_pretty.get(label, 0)])

             tbl = ax3.table(cellText=table_data, colLabels=["Object", "Count"],
                             loc="center", cellLoc='left', colLoc='left')
             tbl.scale(1.05, 1.25)
             for key, cell in tbl.get_celld().items():
                 cell.set_edgecolor('#dddddd')
                 if key[0] == 0:
                     cell.set_facecolor('#f2f2f2'); cell.set_text_props(weight='bold'); cell.set_fontsize(FONT_SIZE_TABLE)
                 else:
                     cell.set_fontsize(FONT_SIZE_TABLE)

             legend_handles = [Patch(facecolor='none', edgecolor=_rgb_tuple_from_bgr(PRETTY2COLOR_BGR[p]), label=p)
                               for p in PRETTY_ORDER]
             ax3.legend(handles=legend_handles, title="Class colors", loc='lower center', bbox_to_anchor=(0.5, -0.05),
                        frameon=True, fontsize=FONT_SIZE_TABLE, title_fontsize=FONT_SIZE_TABLE)

             fig.savefig(fig_buf, format='png', bbox_inches='tight')
             plt.close(fig)
             fig_bytes = fig_buf.getvalue()

             compare_id = save_bytes_to_gridfs(
                 fig_bytes,
                 filename=f"{img_stem}_compare.png",
                 metadata={"inference_id": str(inference_id), "source_image_id": str(gridfs_id), "type": "figure_compare"}
             )

             # Add to results
             results.append({
                 "source_filename": img_name,
                 "source_image_gridfs_id": str(gridfs_id),
                 "artifacts": [
                     {
                         "kind": "nuclei_json",
                         "gridfs_id": str(nuclei_json_id),
                         "filename": f"{img_stem}_nuclei.json",
                     },
                     {
                         "kind": "overlay_stitched",
                         "gridfs_id": str(overlay_id),
                         "filename": f"{img_stem}_stitched.png",
                     },
                     {
                         "kind": "figure_compare",
                         "gridfs_id": str(compare_id),
                         "filename": f"{img_stem}_compare.png",
                     }
                 ]
             })

        # --- Generate Global CSV ---
        if all_rows:
            csv_buf = io.StringIO()
            headers = ["image","nucleus_id","cls_id","cls_name_model","conf","x1","y1","x2","y2"]
            w = csv.DictWriter(csv_buf, fieldnames=headers)
            w.writeheader()
            w.writerows(all_rows)
            csv_bytes = csv_buf.getvalue().encode('utf-8')
            
            csv_id = save_bytes_to_gridfs(
                csv_bytes,
                filename="global_detections.csv",
                metadata={"inference_id": str(inference_id), "type": "csv_global"}
            )
            
            # Add a "dummy" result entry for the global CSV so it appears in the UI/download
            results.append({
                "source_filename": "global_detections.csv",
                "source_image_gridfs_id": None, # No specific source image
                "artifacts": [
                    {
                        "kind": "csv_global",
                        "gridfs_id": str(csv_id),
                        "filename": "global_detections.csv",
                    }
                ]
            })

        # Update inference status
        self.update_inference_status(
            inference_id=inference_id,
            status="completed",
            results=results
        )


