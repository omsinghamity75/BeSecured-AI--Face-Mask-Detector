import argparse
import json
from pathlib import Path

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


DEFAULT_MODEL = Path("artifacts/mask_detector.keras")
DEFAULT_LABELS = Path("artifacts/class_names.json")
DEFAULT_CASCADE = Path("face_detector/haarcascade_frontalface_default.xml")
IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-time face mask detection with a webcam."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def ensure_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, IMAGE_SIZE)
    face = face.astype("float32")
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face


def detect_and_predict_mask(frame, face_cascade, mask_net):
    faces = []
    locs = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    for start_x, start_y, width, height in detections:
        end_x = start_x + width
        end_y = start_y + height
        face = frame[start_y:end_y, start_x:end_x]
        if face.size == 0:
            continue

        faces.append(preprocess_face(face))
        locs.append((start_x, start_y, end_x, end_y))

    if not faces:
        return locs, []

    faces = np.vstack(faces)
    preds = mask_net.predict(faces, verbose=0)
    return locs, preds


def load_class_names(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    ensure_file(args.model, "trained model")
    ensure_file(args.labels, "class labels")
    ensure_file(args.cascade, "Haar cascade face detector")

    class_names = load_class_names(args.labels)
    face_cascade = cv2.CascadeClassifier(str(args.cascade))
    if face_cascade.empty():
        raise RuntimeError(f"Unable to load Haar cascade from {args.cascade}")
    mask_net = load_model(args.model)

    video_stream = VideoStream(src=args.camera).start()
    print("[INFO] Starting video stream. Press 'q' to quit.")

    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                break

            frame = imutils.resize(frame, width=600)
            locs, preds = detect_and_predict_mask(frame, face_cascade, mask_net)

            for (start_x, start_y, end_x, end_y), pred in zip(locs, preds):
                class_index = int(np.argmax(pred))
                label_name = class_names[class_index]
                confidence = float(pred[class_index]) * 100

                label = label_name.replace("_", " ").title()
                color = (0, 255, 0) if "with" in label_name else (0, 0, 255)
                display = f"{label}: {confidence:.2f}%"

                cv2.putText(
                    frame,
                    display,
                    (start_x, max(20, start_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

            cv2.imshow("Face Mask Detection", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        video_stream.stop()


if __name__ == "__main__":
    main()
