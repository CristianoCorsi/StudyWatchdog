"""Alert system — rickroll playback with interruptible control.

Primary alert: plays "Never Gonna Give You Up" when user is distracted too long.
Must be interruptible: stops immediately when studying resumes.
Uses pygame.mixer for audio playback.
"""

import logging
import time

from studywatchdog.config import AlertConfig

logger = logging.getLogger(__name__)


class Alerter:
    """Rickroll alert system with interruptible playback and cooldown.

    Args:
        config: Alert configuration.
    """

    def __init__(self, config: AlertConfig) -> None:
        self._config = config
        self._mixer_initialized = False
        self._is_playing = False
        self._last_alert_time: float = 0.0

    def _ensure_mixer(self) -> bool:
        """Initialize pygame mixer if not already done.

        Returns:
            True if mixer is ready, False if initialization failed.
        """
        if self._mixer_initialized:
            return True

        try:
            import pygame.mixer

            pygame.mixer.init()
            pygame.mixer.music.set_volume(self._config.volume)
            self._mixer_initialized = True
            logger.info("Audio mixer initialized (volume=%.0f%%)", self._config.volume * 100)
            return True
        except Exception as e:
            logger.error("Failed to initialize audio mixer: %s", e)
            return False

    def _check_audio_file(self) -> bool:
        """Check if the rickroll audio file exists.

        Returns:
            True if the file exists.
        """
        if not self._config.rickroll_path.exists():
            logger.warning(
                "Rickroll audio not found: %s — Place an MP3 file there to enable alerts.",
                self._config.rickroll_path,
            )
            return False
        return True

    def play(self) -> None:
        """Start playing the rickroll audio.

        Respects cooldown to prevent spam. Does nothing if already playing
        or if cooldown hasn't expired.
        """
        if self._is_playing:
            return

        # Check cooldown
        now = time.monotonic()
        elapsed_since_last = now - self._last_alert_time
        if self._last_alert_time > 0 and elapsed_since_last < self._config.cooldown:
            remaining = self._config.cooldown - elapsed_since_last
            logger.debug("Rickroll cooldown: %.0fs remaining", remaining)
            return

        if not self._check_audio_file():
            return

        if not self._ensure_mixer():
            return

        try:
            import pygame.mixer

            pygame.mixer.music.load(str(self._config.rickroll_path))
            pygame.mixer.music.play(loops=-1)  # Loop until stopped
            self._is_playing = True
            self._last_alert_time = now
            logger.info("🎵 RICKROLL ACTIVATED! Never gonna give you up...")
        except Exception as e:
            logger.error("Failed to play rickroll: %s", e)

    def stop(self) -> None:
        """Stop the rickroll immediately."""
        if not self._is_playing:
            return

        try:
            import pygame.mixer

            pygame.mixer.music.stop()
            self._is_playing = False
            logger.info("🔇 Rickroll stopped — back to studying!")
        except Exception as e:
            logger.error("Failed to stop rickroll: %s", e)
            self._is_playing = False

    @property
    def is_playing(self) -> bool:
        """Check if rickroll is currently playing."""
        return self._is_playing

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self._mixer_initialized:
            try:
                import pygame.mixer

                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
            self._mixer_initialized = False
            self._is_playing = False
            logger.info("Audio mixer cleaned up")
