"""napari dock widget cho full microplate pipeline.

npe2 manifest trỏ tới `MicroplateWidget` (callable trả widget). Khi click "Run
Pipeline": lấy ảnh từ layer đang chọn → chạy 3 stage → thêm 2 layer (plate
aligned image + wells shapes màu theo class) + thông báo counts + tuỳ chọn save PNG.
"""

from pathlib import Path

import numpy as np
from magicgui import magic_factory

from ._pipeline import CLASS_COLORS, CLASS_NAMES, get_pipeline
from ._viz import draw_result, save_png


def _to_gray_uint8(data):
    """Ép dữ liệu layer napari về uint8 grayscale 2D."""
    arr = np.asarray(data)
    if arr.ndim == 3:                       # (H,W,3) hoặc (H,W,4)
        if arr.shape[-1] >= 3:
            # napari dùng RGB; OpenCV muốn BGR → đảo kênh cho đúng trước gray
            arr = arr[..., :3][..., ::-1]
        arr = arr.mean(axis=-1) if arr.ndim == 3 else arr
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _bgr_to_rgb01(bgr):
    """BGR uint8 (0..255) → RGB float (0..1) cho face_color của napari Shapes."""
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


@magic_factory(
    call_button="Run Pipeline",
    layout="vertical",
    input_layer={"label": "Image layer"},
    save_png={"label": "Save PNG"},
    save_dir={"label": "Save dir", "mode": "d"},
)
def _pipeline_widget(
    viewer: "napari.Viewer",
    input_layer: "napari.layers.Image",
    save_png: bool = False,
    save_dir: Path = Path.home(),
):
    if input_layer is None:
        from napari.utils.notifications import warning
        warning("Chọn 1 image layer chứa ảnh microplate raw trước.")
        return

    from napari.utils.notifications import info

    img_gray = _to_gray_uint8(input_layer.data)

    pipe = get_pipeline()
    out = pipe.run_array(img_gray)

    # ── Layer 1: plate aligned (grayscale image) ──
    # Gỡ layer cũ cùng tên để chạy lại không chồng
    for nm in ("aligned_plate", "wells"):
        if nm in viewer.layers:
            del viewer.layers[nm]
    viewer.add_image(out['plate'], name="aligned_plate", colormap="gray")

    # ── Layer 2: well bbox (shapes), màu theo class ──
    rects = []
    face_colors = []
    for (x1, y1, x2, y2, _), cls in zip(out['boxes'], out['classes']):
        # napari toạ độ theo (row, col) = (y, x)
        rects.append(np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]]))
        name = CLASS_NAMES.get(int(cls), 'NoGrowth')
        face_colors.append(_bgr_to_rgb01(CLASS_COLORS[name]))

    counts = {'Growth': 0, 'NoGrowth': 0, 'NoAgar': 0}
    for cls in out['classes']:
        counts[CLASS_NAMES.get(int(cls), 'NoGrowth')] += 1

    if rects:
        viewer.add_shapes(
            rects,
            shape_type="rectangle",
            face_color=face_colors,
            edge_color="white",
            edge_width=1,
            opacity=0.35,
            name="wells",
        )

    msg = (f"Total wells: {len(out['boxes'])} — "
           f"Growth={counts['Growth']} "
           f"NoGrowth={counts['NoGrowth']} "
           f"NoAgar={counts['NoAgar']}")
    info(msg)
    print(msg)

    if save_png:
        canvas, _ = draw_result(out, show_labels=True)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(str(input_layer.name)).stem or "result"
        out_path = save_dir / f"pipeline_{stem}.png"
        save_png(canvas, out_path)
        info(f"Đã lưu: {out_path}")


# npe2 trỏ tới callable này — magic_factory trả widget mới mỗi lần gọi.
MicroplateWidget = _pipeline_widget
