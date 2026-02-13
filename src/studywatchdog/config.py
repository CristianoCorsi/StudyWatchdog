"""Configuration management using Pydantic models.

All magic numbers and tunable parameters live here.
Supports loading from TOML file.
"""

# TODO: Implement Pydantic config models
# Camera settings:
#   - capture_interval: float = 3.0 (seconds between frames)
#   - camera_index: int = 0
#
# Detection settings:
#   - model_name: str = "google/siglip-base-patch16-224"
#   - text_candidates: list of text descriptions for each state
#   - studying_threshold: float = 0.5
#
# Decision engine settings:
#   - ema_alpha: float = 0.3
#   - distraction_timeout: float = 30.0 (seconds before alert)
#   - recovery_time: float = 5.0 (seconds of study to exit distraction)
#
# Alert settings:
#   - rickroll_path: Path to audio file
#   - cooldown: float = 60.0 (seconds between rickrolls)
