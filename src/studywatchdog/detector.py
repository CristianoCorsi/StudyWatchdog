"""Study activity detection using SigLIP zero-shot image classification.

Uses SigLIP (google/siglip-base-patch16-224) to compute similarity scores
between webcam frames and text descriptions of activities.

Output is numerical scores (0.0-1.0), NOT generated text.
"""

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipModel, SiglipTokenizer

from studywatchdog.config import DetectorConfig

logger = logging.getLogger(__name__)


class ActivityStatus(enum.Enum):
    """Detected activity classification."""

    STUDYING = "studying"
    NOT_STUDYING = "not_studying"
    ABSENT = "absent"


@dataclass
class DetectionResult:
    """Result of a single frame detection.

    Attributes:
        status: Classified activity.
        confidence: Confidence score (0.0-1.0) for the winning class.
        studying_score: Aggregated score for studying candidates.
        not_studying_score: Aggregated score for not-studying candidates.
        absent_score: Aggregated score for absent candidates.
        scores: Per-candidate raw scores.
        inference_ms: Inference time in milliseconds.
    """

    status: ActivityStatus
    confidence: float
    studying_score: float
    not_studying_score: float
    absent_score: float
    scores: dict[str, float] = field(default_factory=dict)
    inference_ms: float = 0.0


class Detector(Protocol):
    """Protocol for study activity detectors (duck typing)."""

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Analyze a frame and return detection result.

        Args:
            frame: BGR image as numpy array (from OpenCV).

        Returns:
            DetectionResult with classification and scores.
        """
        ...

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        ...


class SigLIPDetector:
    """SigLIP-based zero-shot image classifier for study detection.

    Computes cosine similarity between webcam frames and configurable
    text descriptions. Returns numerical scores — no text generation.

    Args:
        config: Detector configuration.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._model: SiglipModel | None = None
        self._tokenizer: SiglipTokenizer | None = None
        self._image_processor: SiglipImageProcessor | None = None
        self._device: torch.device | None = None
        self._text_inputs: dict | None = None
        self._all_candidates: list[str] = []
        self._studying_indices: list[int] = []
        self._not_studying_indices: list[int] = []
        self._absent_indices: list[int] = []

    def _resolve_device(self) -> torch.device:
        """Determine the best available device."""
        if self._config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info("Using CUDA: %s (%.1f GB)", gpu_name, gpu_mem)
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        else:
            device = torch.device(self._config.device)
            logger.info("Using device: %s", device)
        return device

    def load(self) -> None:
        """Load the SigLIP model and pre-compute text embeddings.

        This is called lazily on first detection, not at import time.
        """
        logger.info("Loading SigLIP model: %s ...", self._config.model_name)
        start = time.monotonic()

        self._device = self._resolve_device()

        self._model = SiglipModel.from_pretrained(self._config.model_name).to(self._device)
        self._model.eval()
        self._tokenizer = SiglipTokenizer.from_pretrained(self._config.model_name)
        self._image_processor = SiglipImageProcessor.from_pretrained(self._config.model_name)

        # Build candidate list with tracked indices per category
        self._all_candidates = []
        self._studying_indices = []
        self._not_studying_indices = []
        self._absent_indices = []

        for text in self._config.studying_candidates:
            self._studying_indices.append(len(self._all_candidates))
            self._all_candidates.append(text)

        for text in self._config.not_studying_candidates:
            self._not_studying_indices.append(len(self._all_candidates))
            self._all_candidates.append(text)

        for text in self._config.absent_candidates:
            self._absent_indices.append(len(self._all_candidates))
            self._all_candidates.append(text)

        # Pre-compute text embeddings (they never change)
        self._precompute_text_embeddings()

        elapsed = time.monotonic() - start
        logger.info(
            "SigLIP loaded in %.1fs (%d text candidates)",
            elapsed,
            len(self._all_candidates),
        )

    def _precompute_text_embeddings(self) -> None:
        """Pre-compute and cache text embeddings for all candidates."""
        assert self._tokenizer is not None
        assert self._model is not None

        text_inputs = self._tokenizer(
            self._all_candidates,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            text_output = self._model.get_text_features(**text_inputs)
            self._text_embeds = text_output.pooler_output
            self._text_embeds = self._text_embeds / self._text_embeds.norm(dim=-1, keepdim=True)

        logger.debug("Pre-computed %d text embeddings", len(self._all_candidates))

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Analyze a frame using SigLIP zero-shot classification.

        Args:
            frame: BGR image as numpy array (from OpenCV).

        Returns:
            DetectionResult with per-category scores and classification.
        """
        if not self.is_loaded():
            self.load()

        assert self._image_processor is not None
        assert self._model is not None
        assert self._text_embeds is not None

        start = time.monotonic()

        # Convert BGR (OpenCV) → RGB (PIL)
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)

        # Get image embedding
        image_inputs = self._image_processor(images=pil_image, return_tensors="pt").to(
            self._device
        )

        with torch.no_grad():
            image_output = self._model.get_image_features(**image_inputs)
            image_embeds = image_output.pooler_output
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # Compute similarity scores (dot product of normalized embeddings)
            # SigLIP uses a learned temperature, but for classification we use sigmoid
            logits = (
                torch.matmul(image_embeds, self._text_embeds.t()) * self._model.logit_scale.exp()
                + self._model.logit_bias
            )
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Build per-candidate scores dict
        per_candidate_scores: dict[str, float] = {}
        for i, text in enumerate(self._all_candidates):
            per_candidate_scores[text] = float(probs[i])

        # Aggregate scores per category (max of candidates in each group)
        studying_score = float(max(probs[i] for i in self._studying_indices))
        not_studying_score = float(max(probs[i] for i in self._not_studying_indices))
        absent_score = float(max(probs[i] for i in self._absent_indices))

        # Classify based on highest category score
        category_scores = {
            ActivityStatus.STUDYING: studying_score,
            ActivityStatus.NOT_STUDYING: not_studying_score,
            ActivityStatus.ABSENT: absent_score,
        }
        status = max(category_scores, key=category_scores.get)  # type: ignore[arg-type]
        confidence = category_scores[status]

        inference_ms = (time.monotonic() - start) * 1000

        logger.debug(
            "Detection: %s (%.2f) in %.0fms | study=%.2f distract=%.2f absent=%.2f",
            status.value,
            confidence,
            inference_ms,
            studying_score,
            not_studying_score,
            absent_score,
        )

        return DetectionResult(
            status=status,
            confidence=confidence,
            studying_score=studying_score,
            not_studying_score=not_studying_score,
            absent_score=absent_score,
            scores=per_candidate_scores,
            inference_ms=inference_ms,
        )
