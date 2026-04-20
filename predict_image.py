import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict whether a face crop has a mask from a single image."
    )
    parser.add_argument("image", type=Path, help="Path to the face image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/mask_detector.keras"),
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("artifacts/class_names.json"),
        help="Path to the class-name JSON file.",
    )
    return parser.parse_args()


def ensure_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def preprocess_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32")
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


def main():
    args = parse_args()
    ensure_file(args.image, "input image")
    ensure_file(args.model, "trained model")
    ensure_file(args.labels, "class labels")

    class_names = json.loads(args.labels.read_text(encoding="utf-8"))
    model = load_model(args.model)

    image = preprocess_image(args.image)
    preds = model.predict(image, verbose=0)[0]

    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index]) * 100
    label = class_names[class_index]

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
