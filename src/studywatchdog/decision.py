"""Decision engine — EMA smoothing + Finite State Machine.

Prevents single-frame noise from triggering false alerts.
Uses Exponential Moving Average for score smoothing and
a FSM with temporal tolerance for state transitions.
"""

import enum
import logging
import time

from studywatchdog.config import DecisionConfig
from studywatchdog.detector import ActivityStatus, DetectionResult

logger = logging.getLogger(__name__)


class StudyState(enum.Enum):
    """FSM states for the study monitor."""

    STUDYING = "STUDYING"
    DISTRACTED = "DISTRACTED"
    ALERT_ACTIVE = "ALERT_ACTIVE"


class DecisionEngine:
    """EMA + FSM decision engine for study monitoring.

    Smooths raw detector scores with an Exponential Moving Average,
    then uses a Finite State Machine with time-based transitions to decide
    when to trigger or stop alerts.

    Args:
        config: Decision engine configuration.
    """

    def __init__(self, config: DecisionConfig) -> None:
        self._config = config
        self._state = StudyState.STUDYING
        self._ema_studying: float = 1.0  # Start assuming studying
        self._state_entered_at: float = time.monotonic()
        self._last_detection: DetectionResult | None = None

    @property
    def state(self) -> StudyState:
        """Current FSM state."""
        return self._state

    @property
    def ema_studying(self) -> float:
        """Current EMA smoothed studying score."""
        return self._ema_studying

    @property
    def time_in_state(self) -> float:
        """Seconds since entering the current state."""
        return time.monotonic() - self._state_entered_at

    @property
    def last_detection(self) -> DetectionResult | None:
        """Most recent detection result."""
        return self._last_detection

    def _transition_to(self, new_state: StudyState) -> None:
        """Transition to a new FSM state."""
        old_state = self._state
        self._state = new_state
        self._state_entered_at = time.monotonic()
        logger.info(
            "State: %s -> %s (EMA=%.2f)",
            old_state.value,
            new_state.value,
            self._ema_studying,
        )

    def _compute_studying_ratio(self, result: DetectionResult) -> float:
        """Compute a 0-1 studying ratio from detection scores.

        Maps the raw detection scores to a single float where
        1.0 = definitely studying, 0.0 = definitely not studying.
        Absent is treated as not-studying.

        Args:
            result: Detection result from the detector.

        Returns:
            Studying ratio between 0.0 and 1.0.
        """
        # If absent, treat as not studying
        if result.status == ActivityStatus.ABSENT:
            return 0.0

        # Ratio based on studying vs not-studying scores
        total = result.studying_score + result.not_studying_score
        if total < 0.01:
            return 0.5  # Indeterminate
        return result.studying_score / total

    def update(self, result: DetectionResult) -> StudyState:
        """Process a new detection result and update the FSM.

        Args:
            result: Latest detection result from the detector.

        Returns:
            Current FSM state after processing.
        """
        self._last_detection = result

        # Update EMA
        raw_score = self._compute_studying_ratio(result)
        alpha = self._config.ema_alpha
        self._ema_studying = alpha * raw_score + (1.0 - alpha) * self._ema_studying

        is_studying = self._ema_studying >= self._config.studying_threshold
        now = time.monotonic()
        time_in_state = now - self._state_entered_at

        # FSM transitions
        if self._state == StudyState.STUDYING:
            if not is_studying:
                self._transition_to(StudyState.DISTRACTED)

        elif self._state == StudyState.DISTRACTED:
            if is_studying and time_in_state >= self._config.recovery_time:
                # Recovered: back to studying
                self._transition_to(StudyState.STUDYING)
            elif is_studying:
                # Still recovering, stay distracted but log
                logger.debug(
                    "Recovering... %.1fs / %.1fs",
                    time_in_state,
                    self._config.recovery_time,
                )
            elif not is_studying and time_in_state >= self._config.distraction_timeout:
                # Distracted too long: trigger alert
                self._transition_to(StudyState.ALERT_ACTIVE)

        elif self._state == StudyState.ALERT_ACTIVE and is_studying:
            # Resumed studying: stop alert
            self._transition_to(StudyState.STUDYING)

        return self._state

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self._state = StudyState.STUDYING
        self._ema_studying = 1.0
        self._state_entered_at = time.monotonic()
        self._last_detection = None
        logger.info("Decision engine reset")
