# Face Mask Detection with TensorFlow

This project trains a face-mask classifier in TensorFlow and uses OpenCV to detect faces in either a webcam stream or a single image.

## Project Layout

```text
E:\Face Mask Detection
|-- dataset
|   |-- with_mask
|   `-- without_mask
|-- face_detector
|   `-- haarcascade_frontalface_default.xml
|-- artifacts
|   |-- mask_detector.keras
|   `-- class_names.json
|-- train_mask_detector.py
|-- detect_mask.py
|-- predict_image.py
`-- requirements.txt
```

## 1. Install Dependencies

Install Python 3.10 or 3.11, create a virtual environment, and then install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your machine is using a newer Python release and `pip` only offers TensorFlow `2.20` or `2.21`, that is fine for this project too. The requirements file is pinned for that newer TensorFlow range.

## 2. Prepare the Dataset

Create the following folders and place your images inside them:

- `dataset/with_mask`
- `dataset/without_mask`

The training script expects these exact folder names.

## 3. Add OpenCV Face Detector File

Place this file in `face_detector/`:

- `haarcascade_frontalface_default.xml`

This project uses OpenCV's Haar cascade face detector for webcam inference.

## 4. Train the Model

```powershell
python train_mask_detector.py --dataset dataset --epochs 10
```

Outputs are written to `artifacts/`:

- `mask_detector.keras`
- `class_names.json`
- `training_curves.png`

## 5. Run Webcam Detection

```powershell
python detect_mask.py
```

Press `q` to close the camera window.

## 6. Predict on a Single Image

```powershell
python predict_image.py path\to\face.jpg
```

## Notes

- This workspace did not contain Python, TensorFlow, dataset images, or the OpenCV face detector assets at the time this project was scaffolded.
- Because those runtime dependencies are missing locally, the scripts were written and reviewed for consistency but not executed here.
- If `matplotlib` is missing, training still works and simply skips saving `training_curves.png`.
- A dataset with `with_mask` and `without_mask` images has now been imported into this workspace from the public repository [Karan-Malik/FaceMaskDetector](https://github.com/Karan-Malik/FaceMaskDetector).
- If you want, I can next add a notebook version, a Flask/Streamlit UI, or a dataset download helper.
