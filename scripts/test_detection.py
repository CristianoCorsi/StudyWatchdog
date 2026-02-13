"""Quick test of the fixed detector with live webcam."""

import cv2
import numpy as np
from studywatchdog.config import DetectorConfig
from studywatchdog.detector import SigLIPDetector

det = SigLIPDetector(DetectorConfig())
det.load()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No webcam")
    exit(1)

for i in range(3):
    ret, frame = cap.read()
    if not ret:
        continue
    result = det.detect(frame)
    print(f"\nFrame {i+1}: {result.status.value} (conf={result.confidence:.3f})")
    print(f"  Study={result.studying_score:.3f}  Distract={result.not_studying_score:.3f}  Absent={result.absent_score:.3f}")
    print(f"  Inference: {result.inference_ms:.0f}ms")
    for text, score in sorted(result.scores.items(), key=lambda x: -x[1]):
        print(f"    {score:.4f}  {text}")

cap.release()
