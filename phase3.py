"""
Phase 3 – Real-Time Person Recognition via OCR
===============================================
Pipeline:
  1. YOLO tracks persons in the Phase-2 output video.
  2. EasyOCR reads the name pill drawn by Phase 2 above each person.
  3. Once a name is confirmed for a track, it is locked to that track.
  4. Appearance count increments every time a person (re-)enters the scene.
  5. A summary HUD is drawn in the top-right corner.

Key optimisations for speed
  - OCR runs only every OCR_INTERVAL_FRAMES frames for unidentified tracks.
  - OCR crops a tight ROI (just the name pill), scaled up once.
  - EasyOCR is initialised once; result is cached per track-id.
  - Display is done at native resolution but we skip GUI on headless runs.
"""

import cv2
import os
import re
import time
import difflib
import easyocr
from ultralytics import YOLO


# ============================================================
# CONFIGURATION  (tweak here)
# ============================================================

INPUT_VIDEO  = "/home/enigmatix/Videos/Screencasts/phased2_labeled.webm"
OUTPUT_VIDEO = "/home/enigmatix/Videos/Screencasts/phased3_output.webm"
MODEL_PATH   = "yolo26n.pt"

# How many seconds a person must be absent before the
# NEXT appearance of that person counts as a new appearance.
DISAPPEAR_SECONDS = 1.5

# Run OCR every N frames for tracks whose identity is unknown.
# Lower = more accurate but slower.  Higher = faster but may miss names.
OCR_INTERVAL_FRAMES = 10

# Height of the region above the person's top edge to scan for the name pill.
# Increased so the full pill is always captured even if person is near top.
NAME_ROI_HEIGHT = 100

# Known names (lowercase key -> display name).
# Add / remove names here only.
KNOWN_NAMES = {
    "zain":    "Zain",
    "touqeer": "Touqeer",
    "faiz":    "Faiz",
    "abid":    "Abid",
}

# Per-name bounding-box / pill colors (BGR).
NAME_COLORS = {
    "Zain":    (220, 80,  20),
    "Touqeer": (20,  140, 220),
    "Faiz":    (20,  200, 80),
    "Abid":    (160, 20,  220),
    "Unknown": (100, 100, 100),
}

# Minimum OCR confidence to accept a name.
OCR_CONFIDENCE_THRESH = 0.25

# Fuzzy match cutoff: how similar the OCR text must be to a known name.
# 0.6 means 60% character similarity is enough (catches typos / misreads).
FUZZY_CUTOFF = 0.60

# Font used for all on-screen text.
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICK = 2


# ============================================================
# INITIALISE MODELS
# ============================================================

print("[init] Loading YOLO …")
model = YOLO(MODEL_PATH)

print("[init] Loading EasyOCR (first run downloads model) …")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
print("[init] Ready.\n")


# ============================================================
# OPEN VIDEO
# ============================================================

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {INPUT_VIDEO}")

fps    = cap.get(cv2.CAP_PROP_FPS) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

DISAPPEAR_FRAMES = int(fps * DISAPPEAR_SECONDS)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"VP80"),
    fps,
    (width, height),
)


# ============================================================
# STATE TABLES
# ============================================================

# track_id -> verified display name  (e.g.  3 -> "Zain")
track_identity: dict[int, str] = {}

# track_id -> last frame we ran OCR for this track
last_ocr_frame: dict[int, int] = {}

# Per-person appearance tracking
# { "Zain": {"count": 2, "visible": True, "last_frame": 450} }
people: dict[str, dict] = {
    name: {"count": 0, "visible": False, "last_frame": None}
    for name in KNOWN_NAMES.values()
}


# ============================================================
# HELPERS
# ============================================================

def normalize(text: str) -> str | None:
    """
    Return the canonical display name from OCR text.

    Strategy (in order of priority):
      1. Exact substring match  – fastest, handles clean reads.
      2. Fuzzy match via difflib – handles OCR typos like
         'Tougeer', 'Touqer', 'Zain' read as 'Zoin', etc.
    """
    clean = re.sub(r"[^a-z]", "", text.lower())

    # ---- 1. Exact substring match ----
    for key, display in KNOWN_NAMES.items():
        if key in clean:
            return display

    # ---- 2. Fuzzy match ----
    # Compare the OCR word against each known key.
    keys = list(KNOWN_NAMES.keys())
    matches = difflib.get_close_matches(
        clean, keys, n=1, cutoff=FUZZY_CUTOFF
    )
    if matches:
        return KNOWN_NAMES[matches[0]]

    return None


def ocr_name(frame, x1: int, y1: int, x2: int, y2: int) -> str | None:
    """
    Read the name pill drawn above the person's bounding box.
    Returns a canonical display name or None.
    """
    roi_y1 = max(0, y1 - NAME_ROI_HEIGHT)
    roi_y2 = min(height, y1 + 5)
    roi_x1 = max(0, x1 - 15)
    roi_x2 = min(width, x2 + 15)

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return None

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None

    # Scale up for better OCR accuracy
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    ocr_results = reader.readtext(
        roi,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    )

    best_name, best_conf = None, 0.0
    for _, text, conf in ocr_results:
        # Debug: print every raw OCR read so we can diagnose misreads
        print(f"  [ocr-raw] '{text}'  conf={conf:.2f}")
        name = normalize(text)
        if name and conf > best_conf and conf >= OCR_CONFIDENCE_THRESH:
            best_name, best_conf = name, conf

    return best_name


def draw_box_and_label(frame, name: str, x1: int, y1: int, x2: int, y2: int,
                       count: int):
    """Draw bounding box + name + appearance count for a recognised person."""
    color = NAME_COLORS.get(name, NAME_COLORS["Unknown"])

    # ---- Bounding box ----
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ---- Label pill above the box ----
    label      = f"{name}  |  #{count}"
    font_scale = 0.70
    (tw, th), bl = cv2.getTextSize(label, FONT, font_scale, FONT_THICK)

    pad     = 6
    pill_x1 = x1
    pill_y1 = max(0, y1 - th - pad * 2 - bl)
    pill_x2 = x1 + tw + pad * 2
    pill_y2 = y1

    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
    cv2.putText(
        frame, label,
        (pill_x1 + pad, pill_y2 - pad - bl),
        FONT, font_scale, (255, 255, 255), FONT_THICK, cv2.LINE_AA,
    )


def draw_hud(frame):
    """Draw a semi-transparent summary panel in the top-right corner."""
    lines = ["  APPEARANCES  "]
    for name, info in people.items():
        status = "IN " if info["visible"] else "OUT"
        lines.append(f"  {name:<10} {status}  x{info['count']}")

    # Measure panel
    fs      = 0.55
    thick   = 1
    padding = 10
    line_h  = 22
    max_w   = max(cv2.getTextSize(l, FONT, fs, thick)[0][0] for l in lines)
    pw      = max_w + padding * 2
    ph      = line_h * len(lines) + padding

    px1 = width  - pw - 10
    py1 = 10
    px2 = width  - 10
    py2 = py1 + ph

    # Dark semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Text lines
    for i, line in enumerate(lines):
        y = py1 + padding + (i + 1) * line_h
        color = (0, 220, 255) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (px1 + padding, y),
                    FONT, fs, color, thick, cv2.LINE_AA)


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0
t_start      = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # ---- YOLO tracking ----
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.40,
        verbose=False,
    )
    result = results[0]

    # Names visible in THIS frame
    visible_this_frame: set[str] = set()

    # ---- Process each detected person ----
    if result.boxes is not None and result.boxes.id is not None:
        boxes     = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # ---- Identity lookup / OCR ----
            name = track_identity.get(tid)

            if name is None:
                # Only run OCR every N frames to keep things fast
                last = last_ocr_frame.get(tid, -OCR_INTERVAL_FRAMES)
                if frame_number - last >= OCR_INTERVAL_FRAMES:
                    last_ocr_frame[tid] = frame_number
                    detected = ocr_name(frame, x1, y1, x2, y2)
                    if detected:
                        track_identity[tid] = detected
                        name = detected
                        print(f"[OCR]   Track {tid} -> {name}")

            # ---- Update appearance state ----
            if name and name in people:
                visible_this_frame.add(name)
                person = people[name]

                if not person["visible"]:
                    # Person (re-)entered the scene
                    person["count"] += 1
                    person["visible"] = True
                    print(
                        f"[ENTER] {name} – appearance #{person['count']}"
                    )

                person["last_frame"] = frame_number

                # ---- Draw bounding box + label ----
                draw_box_and_label(
                    frame, name, x1, y1, x2, y2, person["count"]
                )

    # ---- Check disappearances ----
    for name, person in people.items():
        if not person["visible"]:
            continue
        if name in visible_this_frame:
            continue
        if person["last_frame"] is None:
            continue
        if frame_number - person["last_frame"] >= DISAPPEAR_FRAMES:
            person["visible"] = False
            print(f"[LEFT]  {name}")

    # ---- HUD overlay ----
    draw_hud(frame)

    # ---- Write & display ----
    writer.write(frame)
    cv2.imshow("Phase 3 – Real-Time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - t_start
print(f"\n[done] Processed {frame_number} frames in {elapsed:.1f}s  "
      f"({frame_number / elapsed:.1f} fps effective)")


# ============================================================
# FINAL APPEARANCE SUMMARY
# ============================================================

print()
print("=" * 40)
print("  FINAL APPEARANCE SUMMARY")
print("=" * 40)
for name, info in people.items():
    print(f"  {name:<12}  {info['count']} appearance(s)")
print()
print(f"[saved] {OUTPUT_VIDEO}")

