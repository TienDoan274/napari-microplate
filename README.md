# napari-microplate

[![PyPI](https://img.shields.io/pypi/v/napari-microplate.svg)](https://pypi.org/project/napari-microplate)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![napari hub](https://img.shields.io/badge/napari-plugin-microplate-green.svg)](https://napari-hub.org)

A [napari](https://napari.org) plugin for end-to-end analysis of microbiology
**microplate images**: align the plate, detect every well, and classify each
well as **Growth / NoGrowth / NoAgar**.

The plugin runs a three-stage pipeline directly in the napari viewer and
renders the result as overlay layers you can inspect interactively.

```
raw microplate image
   │
   ├──► Stage 1 — Align plate       (EfficientNet-B3 corner regression, grayscale)
   │       4-corner detection → perspective warp → canonical plate
   ├──► Stage 2 — Detect wells      (YOLOv8n, single-channel grayscale, ch=1)
   │       bounding box per well
   └──► Stage 3 — Classify wells    (Random Forest)
           Growth / NoGrowth / NoAgar
```

## Installation

```bash
pip install napari-microplate
```

This installs the plugin and all its pure-Python dependencies, **including
PyTorch (CPU build)**. On first run the model weights (~700 MB total) are
downloaded automatically from the HuggingFace Hub and cached under
`~/.cache/napari_microplate/`.

### GPU acceleration (optional, recommended)

The CPU PyTorch build works but Stage 1 + Stage 2 are noticeably faster on GPU.
To use a CUDA GPU, install a CUDA-enabled PyTorch **before** or **after** the
plugin:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The plugin auto-detects CUDA via `torch.cuda.is_available()`.

## Usage

1. Launch napari:

   ```bash
   napari
   ```

2. Open a raw microplate image (`File → Open file(s)` or drag-and-drop).

3. Start the plugin: **`Plugins → Microplate Pipeline`**.

4. In the dock widget:
   - pick the **Image layer** holding your microplate image,
   - (optional) tick **Save PNG** and choose a **Save dir**,
   - click **Run Pipeline**.

Two new layers are added to the viewer:

| Layer | Type | Content |
|-------|------|---------|
| `aligned_plate` | Image (gray) | Plate after Stage 1 perspective warp |
| `wells` | Shapes (rectangles) | One box per well, colored by predicted class |

Well counts (`Growth / NoGrowth / NoAgar`) are shown in the napari status bar.

## Configuration

| Environment variable | Purpose | Default |
|----------------------|---------|---------|
| `MICROPLATE_HF_REPO` | HuggingFace repo id for the weights | `tiendoan274/napari-microplate-weights` |
| `MICROPLATE_WEIGHTS_DIR` | Local folder of pre-downloaded weights (skip download) | *unset* |

To run fully offline, download the three weight files into a folder and set
`MICROPLATE_WEIGHTS_DIR` to that folder:

```
$MICROPLATE_WEIGHTS_DIR/
  stage1_efficientnet_b3.pt
  stage2_yolov8n_well.pt
  stage3_random_forest.joblib
```

## Programmatic use (without the GUI)

```python
from napari_microplate._pipeline import MicroplatePipeline

pipe = MicroplatePipeline()          # loads weights lazily on first run
result = pipe.run("plate.png")       # path to a raw microplate image

print(result["boxes"])               # [(x1,y1,x2,y2,conf), ...]
print(result["classes"])             # [0, 1, 2, ...]  (0=Growth,1=NoGrowth,2=NoAgar)
```

## License

Copyright (C) 2026 Tien Doan. Distributed under the
**GNU Affero General Public License v3.0 or later (AGPL-3.0+)** — see
[LICENSE](LICENSE).

This package vendors a modified copy of [Ultralytics 8.3.2](https://github.com/ultralytics/ultralytics)
(AGPL-3.0) under `napari_microplate/_vendor/` to support single-channel
grayscale well detection. See [NOTICE](NOTICE) for details.

## Citation

If you use this plugin in published research, please cite the accompanying paper
(see `paper/` in the source repository).
