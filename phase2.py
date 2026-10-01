import cv2
import os
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_VIDEO  = "/home/enigmatix/Videos/Screencasts/output_video.webm"
OUTPUT_VIDEO = "/home/enigmatix/Videos/Screencasts/phased2_labeled.webm"
MODEL_PATH   = "yolo26n.pt"

# YOLO Track ID -> Name mapping
# Phase 3 reads these names via OCR from this output video.
TRACK_TO_NAME = {
    1: "Zain",
    2: "Touqeer",
    3: "Faiz",
    4: "Abid",
}

# ---- Text rendering config ----
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.80
FONT_THICK = 2
TEXT_COLOR = (255, 255, 255)   # white text

# Per-name pill colors (BGR)
NAME_COLORS = {
    "Zain":    (20,  80,  220),
    "Touqeer": (220, 80,  20),
    "Faiz":    (20,  180, 80),
    "Abid":    (140, 20,  220),
    "Unknown": (80,  80,  80),
}


# ============================================================
# SETUP
# ============================================================

os.makedirs("output", exist_ok=True)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

fps    = cap.get(cv2.CAP_PROP_FPS) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"VP80"),
    fps,
    (width, height),
)


# ============================================================
# HELPER  – draw ONLY the name pill (no bounding box)
# ============================================================

def draw_name_pill(frame, name, x1, y1):
    """
    Draw the person's name in a colored pill above their position.
    NO bounding box. NO track ID. Just the clean name label.
    """
    color = NAME_COLORS.get(name, NAME_COLORS["Unknown"])
    (tw, th), baseline = cv2.getTextSize(name, FONT, FONT_SCALE, FONT_THICK)

    pad     = 7
    pill_x1 = x1
    pill_y1 = max(0, y1 - th - pad * 2 - baseline)
    pill_x2 = x1 + tw + pad * 2
    pill_y2 = max(th + pad * 2, y1)

    # Filled background pill
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)

    # White name text inside the pill
    cv2.putText(
        frame, name,
        (pill_x1 + pad, pill_y2 - pad - baseline),
        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA,
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],      # person only
        conf=0.40,
        verbose=False,
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes     = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            name = TRACK_TO_NAME.get(track_id, "Unknown")

            # ------ ONLY NAME PILL, NO BOUNDING BOX ------
            draw_name_pill(frame, name, x1, y1)

    writer.write(frame)
    cv2.imshow("Phase 2 – Name Only", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"[Phase 2] Done. Saved to: {OUTPUT_VIDEO}")