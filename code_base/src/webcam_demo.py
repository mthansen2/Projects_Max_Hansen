from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import mediapipe as mp

from .labels import CLASSES
from .models.factory import build_model


def make_transform(img_size: int):
    # Match validation preprocessing (resize + imagenet normalize)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exps = np.exp(logits)
    return exps / (exps.sum() + 1e-12)


# Simple hand skeleton connections for drawing (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm links
]


def draw_hand_landmarks_cv2(img_bgr: np.ndarray, landmarks, color=(0, 255, 255)):
    """landmarks: list of objects with .x .y normalized to [0,1]."""
    h, w = img_bgr.shape[:2]
    pts = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        pts.append((px, py))
        cv2.circle(img_bgr, (px, py), 3, color, -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(img_bgr, pts[a], pts[b], color, 2)


def bbox_from_task_landmarks(landmarks, w: int, h: int, pad: float = 0.25):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(min(xs) * w)
    y1 = int(min(ys) * h)
    x2 = int(max(xs) * w)
    y2 = int(max(ys) * h)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # pad + make square-ish
    side = int(max(bw, bh) * (1.0 + pad))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, cx + side // 2)
    y2 = min(h, cy + side // 2)

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mirror", action="store_true",
                    help="mirror whole frame (recommended for webcam demos)")

    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--smooth", type=int, default=8, help="moving-average window for probs")

    ap.add_argument("--roi-scale", type=float, default=0.60,
                    help="fallback: fraction of min(frame_h, frame_w) used for centered square ROI")
    ap.add_argument("--draw-roi", action="store_true", help="draw ROI box")
    ap.add_argument("--conf-thresh", type=float, default=0.60, help="below this, show '?'")

    # Hand landmarking (MediaPipe Tasks)
    ap.add_argument("--hand-landmarks", action="store_true",
                    help="use MediaPipe Tasks HandLandmarker; draw landmarks and use hand bbox as ROI")
    ap.add_argument("--mp-model", type=str, default="models/hand_landmarker.task",
                    help="path to MediaPipe hand_landmarker.task")
    ap.add_argument("--hand-max", type=int, default=1)
    ap.add_argument("--hand-det", type=float, default=0.5, help="min_hand_detection_confidence")
    ap.add_argument("--hand-track", type=float, default=0.5, help="min_tracking_confidence")
    ap.add_argument("--hand-pad", type=float, default=0.25, help="bbox padding factor")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[info] device={device}")
    print(f"[info] loading ckpt: {args.ckpt}")

    model = build_model(pretrained=False).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    tfm = make_transform(args.img_size)
    prob_hist = deque(maxlen=max(1, args.smooth))

    # MediaPipe Tasks landmarker (if enabled)
    landmarker = None
    if args.hand_landmarks:
        model_path = Path(args.mp_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at: {model_path}\n"
                "Download it (PowerShell):\n"
                "  mkdir models -Force\n"
                "  Invoke-WebRequest -Uri "
                "\"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task\" "
                "-OutFile \"models/hand_landmarker.task\""
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=args.hand_max,
            min_hand_detection_confidence=args.hand_det,
            min_tracking_confidence=args.hand_track,
        )
        landmarker = HandLandmarker.create_from_options(options)
        print(f"[info] MediaPipe Tasks HandLandmarker enabled (model={model_path})")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    t0 = time.time()
    while not cap.isOpened() and (time.time() - t0) < 8.0:
        time.sleep(0.25)
        cap.release()
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Try --camera 1 or check Windows camera permissions."
        )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps_t0 = time.time()
    fps_frames = 0
    fps = 0.0

    print("[info] press 'q' to quit")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            h, w = frame_bgr.shape[:2]
            out = frame_bgr.copy()

            # Decide ROI: prefer hand bbox if enabled + found
            bbox = None

            if landmarker is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)  # must be monotonic
                res = landmarker.detect_for_video(mp_image, timestamp_ms)

                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    lms = res.hand_landmarks[0]  # first hand
                    draw_hand_landmarks_cv2(out, lms)
                    bbox = bbox_from_task_landmarks(lms, w, h, pad=args.hand_pad)

            if bbox is None:
                # fallback: centered ROI
                side = int(min(h, w) * args.roi_scale)
                cx, cy = w // 2, h // 2
                x1 = max(0, cx - side // 2)
                y1 = max(0, cy - side // 2)
                x2 = min(w, cx + side // 2)
                y2 = min(h, cy + side // 2)
            else:
                x1, y1, x2, y2 = bbox

            roi_bgr = frame_bgr[y1:y2, x1:x2]

            # Inference on ROI
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(roi_rgb)
            x = tfm(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)[0].detach().float().cpu().numpy()

            probs = softmax_np(logits)
            prob_hist.append(probs)
            probs_sm = np.mean(np.stack(prob_hist, axis=0), axis=0)

            top_idx = np.argsort(-probs_sm)[: max(1, args.topk)]
            top = [(CLASSES[i], float(probs_sm[i])) for i in top_idx]

            pred_label, pred_conf = top[0]
            show_label = pred_label if pred_conf >= args.conf_thresh else "?"

            # FPS calc
            fps_frames += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps = fps_frames / dt
                fps_frames = 0
                fps_t0 = time.time()

            # Overlay text
            cv2.putText(out, f"Pred: {show_label} ({pred_conf:.2f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(out, f"FPS: {fps:.1f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Top-k list
            y0 = 120
            for j, (lab, p) in enumerate(top):
                cv2.putText(out, f"{j+1}) {lab:>7s}: {p:.2f}",
                            (20, y0 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if args.draw_roi or (landmarker is not None):
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, "ROI", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("ASL ResNet18 Demo", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
