# 🎥 Video Person Tracking

A multi-phase computer-vision pipeline that **detects, tracks, and identifies people** in video footage using [YOLOv8](https://github.com/ultralytics/ultralytics) and [EasyOCR](https://github.com/JaidedAI/EasyOCR).

---

## 📋 Overview

The pipeline is split into three progressive phases:

| Phase | Script | Description |
|-------|--------|-------------|
| **1** | `label_video.py` | YOLO tracking with bounding boxes + ID/name labels |
| **2** | `phase2.py` | Clean name-pill overlay (no bounding box) for OCR readability |
| **3** | `phase3.py` | Real-time OCR-based recognition, appearance counting, and HUD |

---

## 🏗️ Project Structure

```
video_person_tracking/
├── label_video.py      # Phase 1 – Bounding box + ID label tracking
├── phase2.py           # Phase 2 – Name-pill overlay (OCR-friendly output)
├── phase3.py           # Phase 3 – OCR recognition, appearance counter & HUD
├── yolo26n.pt          # YOLO model weights (not committed – download separately)
├── output/             # Generated output videos (not committed)
└── venv/               # Python virtual environment (not committed)
```

---

## ⚙️ Requirements

- Python 3.10+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://pypi.org/project/opencv-python/)

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install ultralytics easyocr opencv-python
```

---

## 🚀 Usage

### Phase 1 – Bounding Box Tracking

Draws bounding boxes with YOLO track IDs and mapped names on the video.

```bash
python label_video.py
```

> Edit `INPUT_VIDEO` and `TRACK_TO_NAME` inside the script before running.

---

### Phase 2 – Name-Pill Overlay

Produces a clean output with only a colored name pill per person (no bounding box).  
This output is consumed by Phase 3 for OCR-based recognition.

```bash
python phase2.py
```

> Edit `INPUT_VIDEO`, `OUTPUT_VIDEO`, and `TRACK_TO_NAME` in the configuration block.

---

### Phase 3 – Real-Time OCR Recognition

Reads the Phase 2 output, uses EasyOCR to identify names from the colored pills, tracks appearances, and draws a live HUD.

```bash
python phase3.py
```

> Edit `INPUT_VIDEO`, `OUTPUT_VIDEO`, and `KNOWN_NAMES` in the configuration block.

**Key configuration knobs in `phase3.py`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DISAPPEAR_SECONDS` | `1.5` | Seconds absent before counting as a new appearance |
| `OCR_INTERVAL_FRAMES` | `10` | Run OCR every N frames (lower = more accurate, slower) |
| `OCR_CONFIDENCE_THRESH` | `0.25` | Minimum OCR confidence to accept a name |
| `FUZZY_CUTOFF` | `0.60` | Fuzzy-match similarity threshold for OCR typo correction |
| `NAME_ROI_HEIGHT` | `100` | Pixel height of region above person to scan for name pill |

---

## 🗂️ Tracked People

The mapping of YOLO track IDs to real names is defined in each script:

```python
TRACK_TO_NAME = {
    1: "Zain",
    2: "Touqeer",
    3: "Faiz",
    4: "Abid",
}
```

Update this dictionary to match the people in your video.

---

## 🎨 How the Pipeline Works

```
Input Video
    │
    ▼
Phase 1 (label_video.py)
  YOLO tracking → bounding boxes + "Name | ID: N" labels
    │
    ▼
Phase 2 (phase2.py)
  YOLO tracking → colored name pills only (clean, OCR-friendly)
    │
    ▼
Phase 3 (phase3.py)
  YOLO tracking + EasyOCR → identity confirmed per track
  → bounding boxes + name + appearance count + HUD overlay
    │
    ▼
Output Video (phased3_output.webm)
```

---

## ⌨️ Controls

Press **`Q`** in any OpenCV window to stop processing early.

---

## 📁 Output

Output videos are saved to the paths configured in each script (default: `output/` directory or the paths specified in the `OUTPUT_VIDEO` variables). These files are excluded from version control via `.gitignore`.

---

## 📝 Notes

- Model weights (`*.pt`) are **not committed** — place `yolo26n.pt` in the project root before running.
- Phase 3 downloads EasyOCR models on first run (~200 MB).
- GPU is not required; all scripts run on CPU by default.
