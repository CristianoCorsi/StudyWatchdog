"""Configuration management using Pydantic models.

All magic numbers and tunable parameters live here.
Supports loading from TOML file.
"""

import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project root (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


class CameraConfig(BaseModel):
    """Camera capture settings."""

    camera_index: int = Field(default=0, description="Camera device index (0 = default)")
    capture_interval: float = Field(
        default=3.0, ge=0.5, le=30.0, description="Seconds between frame captures"
    )
    frame_width: int = Field(default=640, description="Capture width in pixels")
    frame_height: int = Field(default=480, description="Capture height in pixels")


class DetectorConfig(BaseModel):
    """SigLIP detector settings."""

    model_name: str = Field(
        default="google/siglip-base-patch16-224",
        description="HuggingFace model identifier for SigLIP",
    )
    studying_candidates: list[str] = Field(
        default=[
            "a person studying, reading a book, or working focused at a desk",
            "a person writing notes or working on a laptop with concentration",
            "a person reading a tablet or e-reader with focus",
        ],
        description="Text descriptions that indicate studying",
    )
    not_studying_candidates: list[str] = Field(
        default=[
            "a person distracted, looking at their phone, scrolling social media",
            "a person sleeping, resting their head on the desk",
            "a person looking away from their work, daydreaming",
            "a person eating, drinking, or doing something unrelated to studying",
        ],
        description="Text descriptions that indicate NOT studying",
    )
    absent_candidates: list[str] = Field(
        default=[
            "an empty desk or room with no person visible",
            "an empty chair at a desk, nobody present",
        ],
        description="Text descriptions that indicate nobody is there",
    )
    device: str = Field(
        default="auto",
        description="Device for inference: 'auto', 'cuda', 'cpu'",
    )


class DecisionConfig(BaseModel):
    """Decision engine (EMA + FSM) settings."""

    ema_alpha: float = Field(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="EMA weight for latest score (higher = more reactive)",
    )
    studying_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="EMA score above this = studying",
    )
    distraction_timeout: float = Field(
        default=30.0,
        ge=5.0,
        description="Seconds of distraction before alert triggers",
    )
    recovery_time: float = Field(
        default=5.0,
        ge=1.0,
        description="Seconds of studying to exit distracted state",
    )


class AlertConfig(BaseModel):
    """Alert (rickroll) settings."""

    rickroll_path: Path = Field(
        default=ASSETS_DIR / "rickroll.mp3",
        description="Path to the rickroll audio file",
    )
    cooldown: float = Field(
        default=60.0,
        ge=5.0,
        description="Minimum seconds between rickroll alerts",
    )
    volume: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Playback volume (0.0 to 1.0)",
    )


class AppConfig(BaseModel):
    """Top-level application configuration."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    debug: bool = Field(
        default=False,
        description="Show debug window with camera feed and detection overlay",
    )
    log_level: str = Field(default="INFO", description="Logging level")


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from a TOML file or use defaults.

    Args:
        config_path: Optional path to a TOML config file.

    Returns:
        Validated AppConfig instance.
    """
    if config_path and config_path.exists():
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Support [tool.studywatchdog] section in pyproject.toml
        if "tool" in data and "studywatchdog" in data["tool"]:
            data = data["tool"]["studywatchdog"]

        logger.info("Loaded config from %s", config_path)
        return AppConfig.model_validate(data)

    logger.info("Using default configuration")
    return AppConfig()
