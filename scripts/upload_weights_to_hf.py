"""Push 3 file weights lên HuggingFace Hub (rename sang tên canonical).

Chạy (yêu cầu đã `huggingface-cli login` hoặc set HF_TOKEN):

    python scripts/upload_weights_to_hf.py --repo <user>/napari-microplate-weights
    python scripts/upload_weights_to_hf.py --repo <user>/napari-microplate-weights --private

Sau khi upload xong, sửa DEFAULT_HF_REPO trong napari_microplate/_weights.py
(hoặc set env MICROPLATE_HF_REPO) thành đúng --repo đã dùng.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (đường dẫn local, tên file canonical trên repo)
WEIGHT_MAP = [
    (REPO_ROOT / "stage_1_align_plate" / "best (13).pt",
     "stage1_efficientnet_b3.pt"),
    (REPO_ROOT / "stage_2_detect_well" / "best.pt",
     "stage2_yolov8n_well.pt"),
    (REPO_ROOT / "stage_3_classify_well" / "random_forest_well_classifier_aug.joblib",
     "stage3_random_forest.joblib"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help='HF repo id, vd "tiendoan274/napari-microplate-weights"')
    ap.add_argument("--private", action="store_true",
                    help="Tạo repo private (mặc định public)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Cài huggingface_hub trước: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    url = api.create_repo(repo_id=args.repo, repo_type="model",
                          private=args.private, exist_ok=True)
    print(f"[HF] repo: {url}")

    for local, canonical in WEIGHT_MAP:
        if not local.is_file():
            print(f"  [SKIP] không tìm thấy file local: {local}")
            continue
        size_mb = local.stat().st_size / (1024 * 1024)
        print(f"  [UPLOAD] {local.name} ({size_mb:.1f} MB) -> {canonical}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=canonical,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"          done: {canonical}")

    print(f"\nXong. Kiểm tra: https://huggingface.co/{args.repo}")
    print(f"Sau đó đặt repo id vào env: set MICROPLATE_HF_REPO={args.repo}")
    print("(hoặc sửa DEFAULT_HF_REPO trong napari_microplate/_weights.py)")


if __name__ == "__main__":
    main()
