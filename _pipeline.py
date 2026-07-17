"""Full microplate pipeline: Stage 1 align → Stage 2 detect → Stage 3 classify.

Port từ full_pipeline.ipynb. Đổi: nhận array grayscale trực tiếp (run_array) để
dùng trong napari widget mà không cần ghi file tạm. Models lazy-load 1 lần.
"""

import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn

from ._models import EfficientNetCorner, order_corners_tl_tr_br_bl
from ._weights import resolve_weight

# ── Fork ultralytics ch=1 được VENDOR trong package (napari_microplate/_vendor).
# Đưa _vendor lên đầu sys.path để `import ultralytics` (và các absolute import
# nội bộ của nó) resolve về bản vendor này, không bị nhầm với pip package. ──
_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

# ─────────────────────────────────────────────
# CONFIG — khớp 1:1 với notebook (đừng sửa lẻ tẻ)
# ─────────────────────────────────────────────
STAGE1_VARIANT  = 'b3'
STAGE1_IMG_SIZE = 896

STAGE2_CONF = 0.05
STAGE2_IOU  = 0.15
STAGE2_MAX  = 384

STAGE3_SIZE = 64

# Phải khớp generate_stage2_data.py (PLATE_MARGIN = 0.08).
PLATE_MARGIN = 0.08

CLASS_NAMES = {0: 'Growth', 1: 'NoGrowth', 2: 'NoAgar'}
CLASS_COLORS = {  # BGR
    'Growth':   (0, 180, 0),
    'NoGrowth': (0, 0, 220),
    'NoAgar':   (200, 0, 200),
}


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MicroplatePipeline:
    """Singleton-ish wrapper load 3 model lần đầu, cache lại cho các lần gọi sau."""

    def __init__(self, device=None):
        self.device = device or _get_device()
        self._model1 = None      # EfficientNetCorner
        self._model2 = None      # YOLO
        self._rf_model = None    # RandomForest
        self._w1 = None
        self._w2 = None
        self._w3 = None

    def _ensure_loaded(self):
        if self._model1 is not None:
            return
        print('[MicroplatePipeline] loading models (lần đầu)...')
        self._w1 = resolve_weight('stage1')
        self._w2 = resolve_weight('stage2')
        self._w3 = resolve_weight('stage3')

        # Stage 1
        self._model1 = EfficientNetCorner(variant=STAGE1_VARIANT, dropout=0.0).to(self.device)
        ckpt = torch.load(str(self._w1), map_location=self.device)
        state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
        self._model1.load_state_dict(state)
        self._model1.eval()

        # Stage 2 — import sau khi đã thêm fork lên sys.path
        from ultralytics import YOLO
        self._model2 = YOLO(str(self._w2))

        # Stage 3
        bundle = joblib.load(str(self._w3))
        self._rf_model = bundle['model'] if isinstance(bundle, dict) and 'model' in bundle else bundle

        dev = 'cuda' if self.device.type == 'cuda' else 'cpu'
        print(f'[MicroplatePipeline] ready on {dev}')

    # ── Stage 1 ──────────────────────────────
    @torch.no_grad()
    def _stage1_align_array(self, img_gray):
        """img_gray: 2D uint8. Trả (plate_gray, ordered_corners_px)."""
        h_o, w_o = img_gray.shape[:2]
        img_r = cv2.resize(img_gray, (STAGE1_IMG_SIZE, STAGE1_IMG_SIZE)).astype(np.float32) / 255.0
        img_r = (img_r - 0.485) / 0.229
        t = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0).to(self.device)
        pred = self._model1(t).cpu().numpy().reshape(4, 2)
        corners_px = pred * np.array([w_o, h_o])
        ordered = np.array(order_corners_tl_tr_br_bl(corners_px.tolist()), dtype=np.float32)

        w_top    = np.linalg.norm(ordered[1] - ordered[0])
        w_bottom = np.linalg.norm(ordered[2] - ordered[3])
        h_left   = np.linalg.norm(ordered[3] - ordered[0])
        h_right  = np.linalg.norm(ordered[2] - ordered[1])
        plate_w = int(round((w_top + w_bottom) / 2))
        plate_h = int(round((h_left + h_right) / 2))

        out_w = int(round(plate_w * (1 + 2 * PLATE_MARGIN)))
        out_h = int(round(plate_h * (1 + 2 * PLATE_MARGIN)))
        pad_x = int(plate_w * PLATE_MARGIN)
        pad_y = int(plate_h * PLATE_MARGIN)
        dst = np.array([
            [pad_x,               pad_y],
            [pad_x + plate_w - 1, pad_y],
            [pad_x + plate_w - 1, pad_y + plate_h - 1],
            [pad_x,               pad_y + plate_h - 1],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        plate = cv2.warpPerspective(img_gray, M, (out_w, out_h), borderValue=0)
        return plate, ordered

    # ── Stage 2 ──────────────────────────────
    def _stage2_detect(self, plate_gray):
        H, W = plate_gray.shape[:2]
        res = self._model2(plate_gray, conf=STAGE2_CONF, iou=STAGE2_IOU,
                           max_det=STAGE2_MAX, verbose=False, ch=1)[0]
        boxes = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0])
                boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes

    # ── Stage 3 ──────────────────────────────
    def _stage3_classify(self, plate_gray, boxes, margin_ratio=0.05):
        if not boxes:
            return []
        H, W = plate_gray.shape[:2]
        crops = []
        for (x1, y1, x2, y2, _) in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            half_w = bw * (1 + 2 * margin_ratio) / 2
            half_h = bh * (1 + 2 * margin_ratio) / 2
            x1c, y1c = int(cx - half_w), int(cy - half_h)
            x2c, y2c = int(cx + half_w), int(cy + half_h)
            x1c, y1c = max(0, x1c), max(0, y1c)
            x2c, y2c = min(W, x2c), min(H, y2c)
            crop = (plate_gray[y1c:y2c, x1c:x2c]
                    if x2c > x1c and y2c > y1c
                    else np.zeros((STAGE3_SIZE, STAGE3_SIZE), dtype=np.uint8))
            crop = cv2.resize(crop, (STAGE3_SIZE, STAGE3_SIZE), interpolation=cv2.INTER_AREA)
            crops.append(crop)
        X = np.stack(crops).reshape(-1, STAGE3_SIZE * STAGE3_SIZE).astype(np.float32) / 255.0
        return list(self._rf_model.predict(X))

    # ── Public API ───────────────────────────
    def run_array(self, img_gray):
        """img_gray: 2D uint8 grayscale. Trả dict {plate, corners, boxes, classes}."""
        self._ensure_loaded()
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        plate, corners = self._stage1_align_array(img_gray)
        boxes = self._stage2_detect(plate)
        classes = self._stage3_classify(plate, boxes)
        return {
            'orig': img_gray,
            'corners': corners,
            'plate': plate,
            'boxes': boxes,
            'classes': classes,
        }

    def run(self, img_path):
        """Tiện ích cho CLI/script: đọc file ảnh → run_array."""
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        return self.run_array(img)


# Singleton module-level cho widget dùng chung
_DEFAULT = None


def get_pipeline():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MicroplatePipeline()
    return _DEFAULT
