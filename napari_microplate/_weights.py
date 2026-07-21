"""Resolve đường dẫn weights: HF Hub download + cache, kèm env-var fallback cho dev.

Pattern giống cellpose/SAM: package chỉ chứa code, weights download lần đầu và
cache tại ~/.cache/napari_microplate/. Override bằng:
  - MICROPLATE_HF_REPO       : repo id khác (vd "user/my-weights")
  - MICROPLATE_WEIGHTS_DIR   : thư mục chứa weights canonical (bỏ qua download)

Version check:
  - Plugin tự động so sánh version local vs HF Hub
  - Nếu version khác → tự động download lại weights mới
  - Version lưu trong file VERSION trên HF (vd: v1.0.0)
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
VERSION_FILE = "VERSION"  # File chứa version chung cho cả 3 weights

CACHE_DIR = Path.home() / ".cache" / "napari_microplate"
VERSION_CACHE = CACHE_DIR / "version.txt"


def get_repo_id():
    return os.environ.get("MICROPLATE_HF_REPO", DEFAULT_HF_REPO)


def _get_hf_version():
    """Tải version string từ HF Hub. Trả về None nếu lỗi."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        version_path = hf_hub_download(
            repo_id=get_repo_id(),
            filename=VERSION_FILE,
            cache_dir=str(CACHE_DIR),
            local_dir=str(CACHE_DIR),
            local_dir_use_symlinks=False,
        )
        with open(version_path, 'r') as f:
            return f.read().strip()
    except Exception:
        return None


def _get_cached_version():
    """Đọc version từ cache local. Trả về None nếu chưa có."""
    if VERSION_CACHE.is_file():
        with open(VERSION_CACHE, 'r') as f:
            return f.read().strip()
    return None


def _check_and_update_version():
    """Kiểm tra version HF vs cache. Trả về True nếu cần download lại."""
    hf_version = _get_hf_version()
    if not hf_version:
        return False  # Không có version info trên HF → không check

    cached_version = _get_cached_version()
    if cached_version is None:
        # Chưa có cache → download mới
        VERSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        VERSION_CACHE.write_text(hf_version)
        return True

    if cached_version != hf_version:
        # Version khác → update cache và báo cần download lại
        VERSION_CACHE.write_text(hf_version)
        print(f"[napari-microplate] Weights updated: {cached_version} → {hf_version}")
        return True

    return False  # Version giống nhau → không cần download

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

    Auto-update: nếu version trên HF khác cache → tự động download lại.
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
        # Check version trước khi dùng cache
        if _check_and_update_version():
            # Version đã thay đổi → xóa cache cũ để download lại
            print(f"[napari-microplate] New version available, re-downloading weights...")
            cached.unlink(missing_ok=True)

    # 3) Download từ HF Hub (nếu chưa có cache hoặc vừa xóa)
    if not cached.is_file():
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

    return cached
