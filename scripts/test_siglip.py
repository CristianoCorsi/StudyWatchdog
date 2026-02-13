"""Test SigLIP scoring approaches to find what works for zero-shot classification."""

import numpy as np
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipModel, SiglipTokenizer

model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to("cuda")
model.eval()
tok = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")
proc = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")

texts = [
    "a person studying, reading a book, or working focused at a desk",
    "a person distracted, looking at their phone, scrolling social media",
    "an empty desk or room with no person visible",
]

text_inputs = tok(texts, padding="max_length", return_tensors="pt").to("cuda")
with torch.no_grad():
    text_out = model.get_text_features(**text_inputs)
    text_embeds = text_out.pooler_output
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)


def score_image(img: Image.Image, label: str) -> None:
    """Score an image against all text candidates."""
    img_inputs = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        img_out = model.get_image_features(**img_inputs)
        img_embeds = img_out.pooler_output
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        # Raw cosine similarity (no scale, no bias)
        cosine_sim = torch.matmul(img_embeds, text_embeds.t()).squeeze(0).cpu().numpy()

        # With logit_scale only (no bias)
        scaled = (cosine_sim * model.logit_scale.exp().cpu().numpy())

        # Softmax over candidates (contrastive style)
        softmax_probs = np.exp(scaled) / np.exp(scaled).sum()

        # With both scale + bias (current broken approach)
        with_bias = scaled + model.logit_bias.cpu().numpy()
        sigmoid_probs = 1.0 / (1.0 + np.exp(-with_bias))

    print(f"\n=== {label} ===")
    print(f"{'Text':<65} {'Cosine':>8} {'Softmax':>8} {'Sigmoid':>8}")
    print("-" * 95)
    for i, t in enumerate(texts):
        print(f"  {t:<63} {cosine_sim[i]:>8.4f} {softmax_probs[i]:>8.4f} {sigmoid_probs[i]:>8.4f}")


# Test 1: Black image
score_image(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)), "Black image")

# Test 2: Random noise
score_image(
    Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
    "Random noise",
)

# Test 3: Capture from webcam (if available)
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        rgb = frame[:, :, ::-1]
        score_image(Image.fromarray(rgb), "Live webcam frame")
    cap.release()
else:
    print("\nNo webcam available for live test")
