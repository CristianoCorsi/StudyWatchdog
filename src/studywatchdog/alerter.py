"""Alert system — rickroll playback with interruptible control.

Primary alert: plays "Never Gonna Give You Up" when user is distracted too long.
Must be interruptible: stops immediately when studying resumes.
Supports cooldown to prevent spam.
"""

# TODO: Implement rickroll alert system
# - Use pygame.mixer for audio playback
# - play() / stop() control from decision engine
# - Configurable cooldown between alerts
# - Future: escalation (gentle nudge → rickroll → TTS roast)
# - Audio file in assets/ directory
