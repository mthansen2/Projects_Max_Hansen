# ASL Fingerspelling Recognition (ResNet18 + Webcam Demo)

Real-time ASL alphabet recognition project using a custom ResNet18 classifier in PyTorch, with a webcam demo powered by OpenCV and optional MediaPipe hand landmarks for hand ROI detection.

This project supports:
- **Training / evaluation** of an ASL classifier
- **Real-time webcam inference**
- **Optional hand landmark detection** (MediaPipe Tasks) to improve ROI cropping

## Features

- Custom ResNet18 model (PyTorch)
- Top-k predictions with confidence scores
- Probability smoothing (moving average)
- Webcam ROI fallback (center crop) if no hand is detected
- Optional MediaPipe hand landmarks + hand bounding box
- CPU or CUDA inference

## Project Structure (example)

```text
code_base/
├─ checkpoints/
│  └─ best.pt
├─ models/
│  └─ hand_landmarker.task        # optional (for MediaPipe hand landmarks)
├─ src/
│  ├─ __init__.py
│  ├─ webcam_demo.py
│  ├─ labels.py
│  └─ models/
│     └─ factory.py
├─ requirements.txt
└─ README.md
```

## Requirements

- Python 3.10 or 3.11 recommended
- Webcam
- (Optional) NVIDIA GPU + CUDA-enabled PyTorch
- (Optional) MediaPipe Tasks model file for hand landmarks

## Installation

### 1) Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Verify key installs

```powershell
python -c "import torch, cv2, mediapipe; print('ok')"
```

## Running the Webcam Demo

Because `src/webcam_demo.py` uses relative imports (for example `from .labels import CLASSES`), run it from the project root using `-m`.

### From `code_base/`:

```powershell
python -m src.webcam_demo --ckpt checkpoints/best.pt --mirror
```

### Useful options

```powershell
python -m src.webcam_demo --help
```

Common flags:
- `--mirror` : mirror webcam frame (recommended)
- `--camera 0` : choose camera index (try `1` if needed)
- `--device auto|cpu|cuda`
- `--topk 3`
- `--smooth 8`
- `--draw-roi`
- `--conf-thresh 0.60`

## MediaPipe Hand Landmarks (Optional)

If enabled, the demo:
- detects hand landmarks
- draws the hand skeleton
- uses the detected hand bounding box as the ROI for classification

### Download the MediaPipe hand landmarker model (Windows PowerShell)

```powershell
mkdir models -Force
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models/hand_landmarker.task"
```

### Run with landmarks enabled

```powershell
python -m src.webcam_demo --ckpt checkpoints/best.pt --mirror --hand-landmarks --draw-roi
```

### If landmarks are not appearing

- Make sure you passed `--hand-landmarks`
- Confirm the file exists at `models/hand_landmarker.task`
- Use good lighting and keep your hand centered and visible
- Try lower detection thresholds:

```powershell
python -m src.webcam_demo --ckpt checkpoints/best.pt --mirror --hand-landmarks --hand-det 0.3 --hand-track 0.3 --draw-roi
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
Install dependencies in the same Python environment you are using to run the script:

```powershell
pip install -r requirements.txt
```

### `ImportError: attempted relative import with no known parent package`
Run as a module from project root:

```powershell
python -m src.webcam_demo
```

Not:

```powershell
python .\src\webcam_demo.py
```

### Camera fails to open
- Try a different camera index:
```powershell
python -m src.webcam_demo --camera 1
```
- Check Windows camera permissions (Privacy settings)
- Close other apps using the camera

## Notes

- The webcam demo expects a trained checkpoint (`best.pt`) compatible with your model architecture.
- If your checkpoint was saved as a wrapper dict (e.g. `{"model_state_dict": ...}`), you may need to adjust loading in `webcam_demo.py`.

## Future Improvements (optional ideas)

- Temporal smoothing with exponential moving average
- Per-class confidence calibration
- Better hand ROI tracking
- Support for video file inference
- Export to ONNX / TorchScript

## License

Add a license if you plan to share or reuse publicly (MIT is a common choice).
