"""Study activity detection using SigLIP zero-shot image classification.

Uses SigLIP (google/siglip-base-patch16-224) to compute similarity scores
between webcam frames and text descriptions of activities (studying, distracted, absent).

Output is numerical scores (0.0-1.0), NOT generated text.
"""

# TODO: Implement SigLIP-based detector
# - Load SigLIP model + processor (lazy, on first use)
# - Pre-compute text embeddings at startup (they don't change per frame)
# - detect(frame) → DetectionResult with scores per text candidate
# - Protocol interface for swappable implementations
# - Log inference time for performance monitoring
