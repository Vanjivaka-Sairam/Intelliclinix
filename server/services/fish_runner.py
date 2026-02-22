import sys
import csv
import torch
import cv2
import numpy as np
from pathlib import Path
import math
import io
import json
import pathlib
from collections import defaultdict
from contextlib import contextmanager

from PIL import Image
from bson.objectid import ObjectId
from flask import current_app

from services.model_runner_base import ModelRunner
from services.storage import save_bytes_to_gridfs

# ── YOLOv5 local repo path ──────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
yolov5_path  = project_root / "yolov5"

@contextmanager
def _yolov5_path_ctx():

    yolov5_str = str(yolov5_path)

    # ── 1. Fix sys.path ───────────────────────────────────────────────────────
    original_path = list(sys.path)
    cleaned = [p for p in sys.path if p != yolov5_str]
    sys.path[:] = [yolov5_str] + cleaned

    # ── 2. Evict cached utils modules ─────────────────────────────────────────
    evicted = {}
    for key in list(sys.modules.keys()):
        if key == "utils" or key.startswith("utils."):
            evicted[key] = sys.modules.pop(key)

    # ── 3. Patch PosixPath for models saved on Linux ──────────────────────────
    original_posix = getattr(pathlib, "PosixPath", None)
    pathlib.PosixPath = pathlib.WindowsPath

    try:
        yield
    finally:
        sys.path[:] = original_path
        sys.modules.update(evicted)
        if original_posix is not None:
            pathlib.PosixPath = original_posix


# ── User configs ─────────────────────────────────────────────────────────────
CELL_CLASS_ID   = 4
FUSION_CLASS_ID = 3
GENE_MAP        = {0: "green", 1: "red", 2: "aqua"}

# Human-readable class names for CSV / tooltip
SIGNAL_NAMES = {
    0: "Green",
    1: "Red",
    2: "Aqua",
    FUSION_CLASS_ID: "Fusion",
}

CELL_CONF = 0.5
GENE_CONF = 0.5

# ── Image-processing helpers ─────────────────────────────────────────────────

def yolo_to_cv2(yolo_box, img_shape):
    """Converts YOLO (cx, cy, w, h) normalised → CV2 (x1, y1, x2, y2) pixels."""
    h, w = img_shape[:2]
    cx_n, cy_n, w_n, h_n = yolo_box
    x1 = int((cx_n - w_n / 2) * w)
    y1 = int((cy_n - h_n / 2) * h)
    x2 = int((cx_n + w_n / 2) * w)
    y2 = int((cy_n + h_n / 2) * h)
    return (x1, y1, x2, y2)


def cv2_to_yolo(cv2_box, img_shape, class_id):
    """Converts CV2 pixels → YOLO normalised format with a class ID."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = cv2_box
    bw, bh = x2 - x1, y2 - y1
    return (
        int(class_id),
        max(0, min(1, (x1 + bw / 2) / w)),
        max(0, min(1, (y1 + bh / 2) / h)),
        max(0, min(1, bw / w)),
        max(0, min(1, bh / h)),
    )


def translate_to_global(local_cv2_box, cell_global_cv2_box):
    """Translates coordinates from a local cell patch back to the global image."""
    gx1, gy1, _, _ = cell_global_cv2_box
    lx1, ly1, lx2, ly2 = local_cv2_box
    return (lx1 + gx1, ly1 + gy1, lx2 + gx1, ly2 + gy1)


# ── Fusion helpers ───────────────────────────────────────────────────────────

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_diag(box):
    x1, y1, x2, y2 = box
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def apply_fusion_logic(detections, img_shape):
    """
    Matches Green (0) + Red (1) pairs → Fusion (3).
    Remaining individual signals are kept as-is.
    """
    greens = [d for d in detections if d["cid"] == 0]
    reds   = [d for d in detections if d["cid"] == 1]
    aquas  = [d for d in detections if d["cid"] == 2]

    used_g, used_r = set(), set()
    final_results  = []

    # 1. Match Green–Red pairs (closest first)
    pairs = []
    for i, g in enumerate(greens):
        for j, r in enumerate(reds):
            dist   = get_distance(get_center(g["box"]), get_center(r["box"]))
            thresh = (get_diag(g["box"]) + get_diag(r["box"])) / 2.0
            if dist <= thresh:
                pairs.append((dist, i, j))

    pairs.sort()
    for _, gi, rj in pairs:
        if gi in used_g or rj in used_r:
            continue
        used_g.add(gi)
        used_r.add(rj)

        gc, rc = get_center(greens[gi]["box"]), get_center(reds[rj]["box"])
        fc   = ((gc[0] + rc[0]) / 2, (gc[1] + rc[1]) / 2)
        side = (get_diag(greens[gi]["box"]) + get_diag(reds[rj]["box"])) / 2.828
        f_box = (
            int(fc[0] - side), int(fc[1] - side),
            int(fc[0] + side), int(fc[1] + side),
        )
        final_results.append({"cid": FUSION_CLASS_ID, "box": f_box})

    # 2. Remaining individual signals
    for i, g in enumerate(greens):
        if i not in used_g:
            final_results.append(g)
    for i, r in enumerate(reds):
        if i not in used_r:
            final_results.append(r)
    for a in aquas:
        final_results.append(a)

    return final_results


# ── Nuclei JSON generation ───────────────────────────────────────────────────

def _bbox_to_polygon(x1, y1, x2, y2):
    """Convert a bounding box to a 4-point polygon list [[x,y], ...]."""
    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)],
    ]


def _signal_in_cell(signal_box, cell_box):
    """Return True if the centre of signal_box falls inside cell_box."""
    gx1, gy1, gx2, gy2 = signal_box
    cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
    nx1, ny1, nx2, ny2 = cell_box
    return nx1 <= cx <= nx2 and ny1 <= cy <= ny2


def generate_nuclei_json(nucleus_detections, gene_detections, img_shape, image_id):
    """
    Build the interactive nuclei JSON consumed by the frontend viewer.

    Parameters
    ----------
    nucleus_detections : list of (x1, y1, x2, y2) pixel boxes
    gene_detections    : list of {"cid": int, "box": (x1,y1,x2,y2)} in global coords
    img_shape          : (h, w) of the full image
    image_id           : str identifier (e.g. stem name)
    """
    nuclei_list = []

    for nucleus_id, cell_box in enumerate(nucleus_detections, start=1):
        nx1, ny1, nx2, ny2 = cell_box
        x1 = max(0, nx1)
        y1 = max(0, ny1)
        x2 = min(img_shape[1], nx2)
        y2 = min(img_shape[0], ny2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Count signals inside this cell
        stats = {name: 0 for name in SIGNAL_NAMES.values()}
        for det in gene_detections:
            if _signal_in_cell(det["box"], (x1, y1, x2, y2)):
                name = SIGNAL_NAMES.get(det["cid"])
                if name:
                    stats[name] += 1

        nuclei_list.append({
            "id": nucleus_id,
            "polygon": _bbox_to_polygon(x1, y1, x2, y2),
            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            "stats": stats,
        })

    return {
        "image_id": image_id,
        "nuclei": nuclei_list,
    }


# ── GridFS helpers ───────────────────────────────────────────────────────────

def _load_image_from_gridfs(fs, gridfs_id) -> np.ndarray:
    """Read a GridFS file and return a BGR numpy array."""
    file_bytes = fs.get(gridfs_id).read()
    pil_img    = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _get_channel(files, keyword: str):
    """Return the file-ref whose filename contains the keyword (case-insensitive)."""
    kw = keyword.upper()
    for f in files:
        if kw in f["filename"].upper():
            return f
    return None


# ── Runner ───────────────────────────────────────────────────────────────────

class FishRunner(ModelRunner):

    def run_inference_job(self, inference_id_str: str) -> None:
        inference_id = ObjectId(inference_id_str)

        self.db.inferences.update_one(
            {"_id": inference_id},
            {"$set": {"status": "running"}}
        )

        inference_doc = self.db.inferences.find_one({"_id": inference_id})
        dataset_doc   = self.db.datasets.find_one({"_id": inference_doc["dataset_id"]})

        # ── Load model weights from app config ───────────────────────────────
        nuclei_path_str = current_app.config.get("FISH_NUCLEI_MODEL_PATH")
        gene_path_str   = current_app.config.get("FISH_GENE_MODEL_PATH")

        if not nuclei_path_str or not gene_path_str:
            raise ValueError(
                "FISH model paths not configured. "
                "Set FISH_NUCLEI_MODEL_PATH and FISH_GENE_MODEL_PATH in .env"
            )

        nuclei_weights = Path(nuclei_path_str)
        gene_weights   = Path(gene_path_str)

        if not nuclei_weights.exists():
            raise FileNotFoundError(f"Nuclei weights not found: {nuclei_weights}")
        if not gene_weights.exists():
            raise FileNotFoundError(f"Gene weights not found: {gene_weights}")

        # ── Load models via local YOLOv5 repo ────────────────────────────────
        print(f"[FishRunner] Loading nuclei model: {nuclei_weights}")
        with _yolov5_path_ctx():
            nuclei_model = torch.hub.load(
                str(yolov5_path), "custom",
                path=str(nuclei_weights), source="local"
            )
        nuclei_model.conf = CELL_CONF

        print(f"[FishRunner] Loading gene model: {gene_weights}")
        with _yolov5_path_ctx():
            gene_model = torch.hub.load(
                str(yolov5_path), "custom",
                path=str(gene_weights), source="local", force_reload=True
            )
        gene_model.conf = GENE_CONF

        # ── Load the 5 channel images from GridFS ────────────────────────────
        files = dataset_doc.get("files", [])

        dapi_ref   = _get_channel(files, "DAPI")
        fitc_ref   = _get_channel(files, "FITC")
        orange_ref = _get_channel(files, "ORANGE")
        aqua_ref   = _get_channel(files, "AQUA")
        sky_ref    = _get_channel(files, "SKY")

        if not dapi_ref:
            raise ValueError("No DAPI image found in dataset files.")

        dapi_bgr = _load_image_from_gridfs(self.fs, dapi_ref["gridfs_id"])
        h, w, _  = dapi_bgr.shape
        dapi_rgb = cv2.cvtColor(dapi_bgr, cv2.COLOR_BGR2RGB)

        # SKY as visualisation canvas (black fallback if missing)
        sky_vis = (
            _load_image_from_gridfs(self.fs, sky_ref["gridfs_id"])
            if sky_ref else np.zeros((h, w, 3), dtype=np.uint8)
        )

        # Gene channel map
        channel_imgs = {}
        for label, ref in [("FITC", fitc_ref), ("ORANGE", orange_ref), ("AQUA", aqua_ref)]:
            if ref:
                channel_imgs[label] = _load_image_from_gridfs(self.fs, ref["gridfs_id"])

        # ── Run pipeline ─────────────────────────────────────────────────────
        gene_class_colors = {
            0: (0, 255, 0),                  # Green  (FITC)
            1: (0, 165, 255),                # Orange (ORANGE)
            2: (255, 255, 0),                # Cyan   (AQUA)
            FUSION_CLASS_ID: (255, 0, 255),  # Magenta (Fusion)
        }

        global_labels    = []
        nucleus_boxes    = []   # list of (x1,y1,x2,y2) pixel coords for each detected cell
        all_gene_dets    = []   # list of {"cid": int, "box": (x1,y1,x2,y2)} global coords

        nuclei_results = nuclei_model(dapi_rgb)

        for det in nuclei_results.xywhn[0]:
            nucleus_yolo    = (CELL_CLASS_ID, *det[:4].tolist())
            global_labels.append(nucleus_yolo)

            cell_box_pixels = yolo_to_cv2(det[:4].tolist(), (h, w))
            nx1, ny1, nx2, ny2 = cell_box_pixels
            nucleus_boxes.append(cell_box_pixels)

            # Draw nucleus boundary — white, thickness 2
            cv2.rectangle(sky_vis, (nx1, ny1), (nx2, ny2), (255, 255, 255), 2)

            local_gene_dets  = []
            last_patch_shape = None

            for ch_bgr in channel_imgs.values():
                patch = ch_bgr[max(0, ny1):min(h, ny2), max(0, nx1):min(w, nx2)]
                if patch.size == 0:
                    continue

                last_patch_shape = patch.shape
                patch_rgb        = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                gene_results     = gene_model(patch_rgb)

                for g_det in gene_results.xywhn[0]:
                    g_box_px = yolo_to_cv2(g_det[:4].tolist(), patch.shape)
                    local_gene_dets.append({"cid": int(g_det[5]), "box": g_box_px})

            if local_gene_dets and last_patch_shape:
                fused_dets = apply_fusion_logic(local_gene_dets, last_patch_shape)
                for f_det in fused_dets:
                    global_px   = translate_to_global(f_det["box"], cell_box_pixels)
                    global_yolo = cv2_to_yolo(global_px, (h, w), f_det["cid"])
                    global_labels.append(global_yolo)

                    # Store for nuclei JSON enrichment
                    all_gene_dets.append({"cid": f_det["cid"], "box": global_px})

                    gx1, gy1, gx2, gy2 = global_px
                    color = gene_class_colors.get(f_det["cid"], (0, 0, 255))
                    cv2.rectangle(sky_vis, (gx1, gy1), (gx2, gy2), color, 1)

        # ── Save artifacts to GridFS ──────────────────────────────────────────
        stem = Path(dapi_ref["filename"]).stem.replace("_DAPI", "")

        # 1. DAPI source image (raw, used as the interactive base layer)
        dapi_enc = cv2.imencode(".png", dapi_bgr)[1].tobytes()
        dapi_id  = save_bytes_to_gridfs(
            dapi_enc,
            filename=f"{stem}_DAPI.png",
            metadata={"inference_id": str(inference_id), "type": "dapi_source"}
        )

        # 2. SKY visualisation PNG (used as overlay on top of DAPI)
        vis_enc = cv2.imencode(".png", sky_vis)[1].tobytes()
        vis_id  = save_bytes_to_gridfs(
            vis_enc,
            filename=f"{stem}_SKY_VIS.png",
            metadata={"inference_id": str(inference_id), "type": "sky_vis"}
        )

        # 3. Nuclei JSON — polygon (from bbox corners) + per-nucleus signal stats
        nuclei_data = generate_nuclei_json(nucleus_boxes, all_gene_dets, (h, w), stem)
        nuclei_json_bytes = json.dumps(nuclei_data).encode("utf-8")
        nuclei_json_id = save_bytes_to_gridfs(
            nuclei_json_bytes,
            filename=f"{stem}_nuclei.json",
            metadata={"inference_id": str(inference_id), "type": "nuclei_json"}
        )

        # 4. YOLO label TXT (for reference/export)
        label_lines = [
            f"{int(lbl[0])} " + " ".join(f"{x:.6f}" for x in lbl[1:])
            for lbl in global_labels
        ]
        label_bytes = "\n".join(label_lines).encode("utf-8")
        label_id = save_bytes_to_gridfs(
            label_bytes,
            filename=f"{stem}_SKY.txt",
            metadata={"inference_id": str(inference_id), "type": "yolo_labels"}
        )

        # 5. Global CSV — one row per nucleus with signal counts
        csv_id = None
        if nuclei_data["nuclei"]:
            signal_cols = list(SIGNAL_NAMES.values())  # ["Green", "Red", "Aqua", "Fusion"]
            csv_buf = io.StringIO()
            headers = ["image", "nucleus_id"] + signal_cols
            w_csv = csv.DictWriter(csv_buf, fieldnames=headers, extrasaction="ignore")
            w_csv.writeheader()
            for nucleus in nuclei_data["nuclei"]:
                row = {"image": stem, "nucleus_id": nucleus["id"]}
                for col in signal_cols:
                    row[col] = nucleus["stats"].get(col, 0)
                w_csv.writerow(row)

            csv_bytes = csv_buf.getvalue().encode("utf-8")
            csv_id = save_bytes_to_gridfs(
                csv_bytes,
                filename="global_detections.csv",
                metadata={"inference_id": str(inference_id), "type": "csv_global"}
            )

        # ── Build results list ───────────────────────────────────────────────
        main_artifacts = [
            {
                "kind": "overlay_stitched",    # sky_vis PNG shown as toggleable overlay
                "gridfs_id": str(vis_id),
                "filename": f"{stem}_SKY_VIS.png",
            },
            {
                "kind": "nuclei_json",
                "gridfs_id": str(nuclei_json_id),
                "filename": f"{stem}_nuclei.json",
            },
            {
                "kind": "yolo_labels",
                "gridfs_id": str(label_id),
                "filename": f"{stem}_SKY.txt",
            },
        ]

        results = [
            {
                "source_filename": dapi_ref["filename"],
                "source_image_gridfs_id": str(dapi_id),   # DAPI as base image
                "source_filenames": [f["filename"] for f in files],
                "artifacts": main_artifacts,
            }
        ]

        # Global CSV as a separate result entry (same pattern as D-DISH)
        if csv_id:
            results.append({
                "source_filename": "global_detections.csv",
                "source_image_gridfs_id": None,
                "artifacts": [
                    {
                        "kind": "csv_global",
                        "gridfs_id": str(csv_id),
                        "filename": "global_detections.csv",
                    }
                ],
            })

        self.update_inference_status(
            inference_id=inference_id,
            status="completed",
            results=results,
        )
