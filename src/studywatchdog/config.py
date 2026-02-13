"""Configuration management using Pydantic models.

All magic numbers and tunable parameters live here.
Supports loading from TOML file and generating a default config.
"""

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project root (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# XDG config directory
_XDG_CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
CONFIG_DIR = _XDG_CONFIG_HOME / "studywatchdog"
CONFIG_FILENAME = "config.toml"


class CameraConfig(BaseModel):
    """Camera capture settings."""

    camera_index: int = Field(default=4, description="Camera device index (4 = default)")
    capture_interval: float = Field(
        default=1.0, ge=0.5, le=30.0, description="Seconds between frame captures"
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
            "a person reading a book at a desk",
            "a person writing notes with a pen",
            "a person focused on a laptop screen, working",
            "a student studying at a desk",
        ],
        description="Text descriptions that indicate studying",
    )
    not_studying_candidates: list[str] = Field(
        default=[
            "a person looking at a smartphone",
            "a person sleeping with head on desk",
            "a person yawning or stretching",
            "a person turned away from the desk",
        ],
        description="Text descriptions that indicate NOT studying",
    )
    absent_candidates: list[str] = Field(
        default=[
            "an empty room with no people",
            "an empty chair, nobody sitting",
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


def _find_config_file() -> Path | None:
    """Search for a config file in standard locations.

    Search order:
        1. ./studywatchdog.toml (current directory)
        2. $XDG_CONFIG_HOME/studywatchdog/config.toml (~/.config/studywatchdog/config.toml)

    Returns:
        Path to the first config file found, or None.
    """
    candidates = [
        Path.cwd() / "studywatchdog.toml",
        CONFIG_DIR / CONFIG_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from a TOML file or use defaults.

    If no explicit path is given, searches standard locations automatically.

    Args:
        config_path: Optional path to a TOML config file.

    Returns:
        Validated AppConfig instance.
    """
    # Auto-discover config if not explicitly provided
    if config_path is None:
        config_path = _find_config_file()

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


# ── Default config TOML template ──

_DEFAULT_CONFIG_TOML = '''\
# ============================================================================
# StudyWatchdog — Configuration File
# ============================================================================
#
# This file controls all parameters of the StudyWatchdog application.
# Place it in one of these locations (searched in order):
#
#   1. ./studywatchdog.toml           (current directory)
#   2. ~/.config/studywatchdog/config.toml  (XDG standard)
#
# Or pass it explicitly:  studywatchdog --config /path/to/config.toml
#
# Lines starting with # are comments. Uncomment and edit to override defaults.
# ============================================================================


# ── Camera ──────────────────────────────────────────────────────────────────
[camera]

# Which camera device to use (integer index).
# Use `studywatchdog --list-cameras` to see available devices.
camera_index = 4

# How often (in seconds) a frame is captured for AI analysis.
# Lower = more responsive but uses more GPU. Range: 0.5 – 30.0
capture_interval = 1.0

# Camera resolution (the camera may use the closest supported size).
frame_width = 640
frame_height = 480


# ── Detector (SigLIP AI Model) ─────────────────────────────────────────────
[detector]

# HuggingFace model identifier. The default is a lightweight model (~400MB)
# that runs well on consumer GPUs. Change only if you know what you're doing.
model_name = "google/siglip-base-patch16-224"

# Device for AI inference: "auto" (recommended), "cuda", or "cpu".
# "auto" picks CUDA if available, otherwise CPU.
device = "auto"

# Text candidates for zero-shot classification.
# SigLIP compares the webcam frame against these descriptions and scores
# how well each one matches. You can add, remove, or rewrite them to
# better fit your setup (e.g. your desk, lighting, typical posture).
#
# TIP: Short, concrete descriptions work best. Avoid vague or abstract phrases.

# Descriptions that mean "the user IS studying / working"
studying_candidates = [
    "a person reading a book at a desk",
    "a person writing notes with a pen",
    "a person focused on a laptop screen, working",
    "a student studying at a desk",
]

# Descriptions that mean "the user is NOT studying"
not_studying_candidates = [
    "a person looking at a smartphone",
    "a person sleeping with head on desk",
    "a person yawning or stretching",
    "a person turned away from the desk",
]

# Descriptions that mean "nobody is at the desk"
absent_candidates = [
    "an empty room with no people",
    "an empty chair, nobody sitting",
]


# ── Decision Engine (EMA + Finite State Machine) ───────────────────────────
[decision]

# EMA (Exponential Moving Average) smoothing factor.
# Controls how quickly the system reacts to changes:
#   - Higher (e.g. 0.7) = reacts faster, but more sensitive to noise
#   - Lower  (e.g. 0.1) = reacts slower, but very stable
# Range: 0.01 – 1.0
ema_alpha = 0.3

# EMA score threshold: above this value = "studying", below = "distracted".
# Range: 0.0 – 1.0
studying_threshold = 0.5

# How many seconds of continuous distraction before the rickroll triggers.
# Minimum: 5.0
distraction_timeout = 30.0

# How many seconds of studying needed to exit "distracted" state and go
# back to "studying". Prevents flickering on brief glances at the screen.
# Minimum: 1.0
recovery_time = 5.0


# ── Alert (Rickroll) ───────────────────────────────────────────────────────
[alert]

# Path to the rickroll audio file (MP3 or OGG).
# rickroll_path = "/path/to/rickroll.mp3"

# Minimum seconds between consecutive rickroll alerts (anti-spam).
# Minimum: 5.0
cooldown = 60.0

# Playback volume (0.0 = mute, 1.0 = max).
volume = 0.8


# ── General ─────────────────────────────────────────────────────────────────

# Show the debug window with camera feed, scores, and controls.
debug = false

# Logging level: DEBUG, INFO, WARNING, ERROR
log_level = "INFO"
'''


def generate_default_config(output_path: Path | None = None) -> Path:
    """Write the default config file with full documentation.

    Args:
        output_path: Where to write. Defaults to XDG config dir.

    Returns:
        Path where the file was written.
    """
    if output_path is None:
        output_path = CONFIG_DIR / CONFIG_FILENAME

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_DEFAULT_CONFIG_TOML, encoding="utf-8")
    return output_path
