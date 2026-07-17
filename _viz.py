"""Vẽ kết quả pipeline (overlay BGR) + lưu PNG. Port từ full_pipeline.ipynb."""

import cv2
import numpy as np

from ._pipeline import CLASS_COLORS, CLASS_NAMES


def draw_result(out, alpha_fill=0.15, box_thickness=2, show_labels=True,
                min_box_size_for_label=18, show_index=False):
    """Tạo canvas BGR: plate aligned + well bbox color-coded.

    `out`: dict trả về từ MicroplatePipeline.run_array().
    Trả (canvas_bgr, counts_dict).
    """
    plate_gray = out['plate']
    canvas = cv2.cvtColor(plate_gray, cv2.COLOR_GRAY2BGR)
    overlay = canvas.copy()

    counts = {'Growth': 0, 'NoGrowth': 0, 'NoAgar': 0}

    for idx, ((x1, y1, x2, y2, conf), cls) in enumerate(zip(out['boxes'], out['classes'])):
        name = CLASS_NAMES.get(int(cls), str(cls))
        color = CLASS_COLORS[name]
        counts[name] = counts.get(name, 0) + 1

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

        tick = max(3, int(min(x2 - x1, y2 - y1) * 0.15))
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(canvas, (cx, cy), (cx + dx * tick, cy), (255, 255, 255), 1)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * tick), (255, 255, 255), 1)

        bw, bh = x2 - x1, y2 - y1
        if show_labels and min(bw, bh) >= min_box_size_for_label:
            label = f'{name[:2]} {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            ly = max(th + 2, y1)
            cv2.rectangle(canvas, (x1, ly - th - 3), (x1 + tw + 2, ly), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x1 + 1, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        if show_index:
            cv2.putText(canvas, str(idx), (x1 + 1, y2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1, cv2.LINE_AA)

    canvas = cv2.addWeighted(overlay, alpha_fill, canvas, 1 - alpha_fill, 0)
    return canvas, counts


def save_png(canvas_bgr, path):
    """Lưu canvas BGR ra PNG (BGR→RGB swap KHÔNG cần — imwrite nhận BGR)."""
    ok = cv2.imwrite(str(path), canvas_bgr)
    if not ok:
        raise IOError(f'Không ghi được ảnh ra {path}')
    return path
