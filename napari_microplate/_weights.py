"""Resolve đường dẫn weights: HF Hub download + cache, kèm env-var fallback cho dev.

Pattern giống cellpose/SAM: package chỉ chứa code, weights download lần đầu và
cache tại ~/.cache/napari_microplate/. Override bằng:
  - MICROPLATE_HF_REPO       : repo id khác (vd "user/my-weights")
  - MICROPLATE_WEIGHTS_DIR   : thư mục chứa weights canonical (bỏ qua download)
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — sửa HF_REPO_ID 1 lần sau khi upload (hoặc set env MICROPLATE_HF_REPO)
# ─────────────────────────────────────────────
DEFAULT_HF_REPO = "tiendoan/napari-microplate-weights"

# Tên file canonical trên HF repo (KHÔNG dùng space — host không thân thiện).
WEIGHT_FILES = {
    "stage1": "stage1_efficientnet_b3.pt",
    "stage2": "stage2_yolov8n_well.pt",
    "stage3": "stage3_random_forest.joblib",
}

CACHE_DIR = Path.home() / ".cache" / "napari_microplate"


def get_repo_id():
    return os.environ.get("MICROPLATE_HF_REPO", DEFAULT_HF_REPO)


def resolve_weight(stage):
    """Trả về Path local tới file weight của `stage` ('stage1'|'stage2'|'stage3').

    Thứ tự ưu tiên:
      1. MICROPLATE_WEIGHTS_DIR/<canonical>  (dev fallback, offline)
      2. CACHE_DIR/<canonical>               (đã download trước đó)
      3. hf_hub_download                     (download + cache lần đầu)
    """
    if stage not in WEIGHT_FILES:
        raise ValueError(f"stage phải là một trong {list(WEIGHT_FILES)}, nhận '{stage}'")
    canonical = WEIGHT_FILES[stage]

    # 1) Dev fallback — thư mục local chứa weights canonical
    local_dir = os.environ.get("MICROPLATE_WEIGHTS_DIR")
    if local_dir:
        p = Path(local_dir) / canonical
        if p.is_file():
            return p

    # 2) Cache đã có
    cached = CACHE_DIR / canonical
    if cached.is_file():
        return cached

    # 3) Download từ HF Hub
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Cần cài huggingface_hub để download weights: pip install huggingface_hub"
        ) from e

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # hf_hub_download lưu vào cache_dir theo cấu trúc nội bộ; ta chỉ quan tâm
        # đường dẫn trả về. Đặt local-dir dùng {CACHE_DIR} để cố định vị trí.
        path = hf_hub_download(
            repo_id=get_repo_id(),
            filename=canonical,
            cache_dir=str(CACHE_DIR),
            local_dir=str(CACHE_DIR),
        )
    except Exception as e:
        raise RuntimeError(
            f"Không thể download '{canonical}' từ HF repo '{get_repo_id()}'.\n"
            f"Kiểm tra: đã upload weights? repo đúng? mạng/token hợp lệ?\n"
            f"Lỗi: {e}\n"
            f"Để chạy offline, set MICROPLATE_WEIGHTS_DIR=<thư mục chứa {canonical}>."
        ) from e

    return Path(path)
