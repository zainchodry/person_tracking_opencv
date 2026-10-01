import cv2
import os
from ultralytics import YOLO


INPUT_VIDEO = "/home/enigmatix/Videos/Screencasts/output_video.webm"
OUTPUT_VIDEO = "output/phase1_labeled.webm"

MODEL_PATH = "yolo26n.pt"


# YOLO Track ID -> Name
TRACK_TO_NAME = {
    1: "Zain",
    2: "Touqeer",
    3: "Faiz",
    4: "Abid",
}


os.makedirs("output", exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {INPUT_VIDEO}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*"VP80")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height),
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.40,
        verbose=False,
    )

    result = results[0]

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, track_id in zip(
            boxes,
            track_ids
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # Get name from Track ID
            name = TRACK_TO_NAME.get(
                track_id,
                "Unknown"
            )

            # Person bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Label
            label = f"{name} | ID: {track_id}"

            text_x = x1
            text_y = max(y1 - 10, 30)

            (
                (text_width, text_height),
                baseline,
            ) = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )

            # Text background
            cv2.rectangle(
                frame,
                (
                    text_x,
                    text_y - text_height - baseline,
                ),
                (
                    text_x + text_width,
                    text_y + baseline,
                ),
                (0, 255, 0),
                -1,
            )

            # Name
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    writer.write(frame)

    cv2.imshow(
        "Phase 1 - Person Tracking",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
writer.release()
cv2.destroyAllWindows()

print(
    f"Video saved to: {OUTPUT_VIDEO}"
)