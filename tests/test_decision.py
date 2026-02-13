"""Tests for the decision engine (EMA + FSM)."""

from unittest.mock import patch

from studywatchdog.config import DecisionConfig
from studywatchdog.decision import DecisionEngine, StudyState
from studywatchdog.detector import ActivityStatus, DetectionResult


def _make_result(
    studying: float = 0.8,
    not_studying: float = 0.1,
    absent: float = 0.05,
) -> DetectionResult:
    """Create a DetectionResult with given scores."""
    scores = {ActivityStatus.STUDYING: studying, ActivityStatus.NOT_STUDYING: not_studying}
    status = max(scores, key=scores.get)  # type: ignore[arg-type]
    return DetectionResult(
        status=status,
        confidence=max(studying, not_studying, absent),
        studying_score=studying,
        not_studying_score=not_studying,
        absent_score=absent,
    )


class TestEMA:
    """Test EMA score smoothing."""

    def test_starts_at_one(self) -> None:
        engine = DecisionEngine(DecisionConfig())
        assert engine.ema_studying == 1.0

    def test_ema_updates_with_alpha(self) -> None:
        config = DecisionConfig(ema_alpha=0.5)
        engine = DecisionEngine(config)

        result = _make_result(studying=0.0, not_studying=0.9)
        engine.update(result)

        # EMA = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert abs(engine.ema_studying - 0.5) < 0.05

    def test_ema_smooths_noise(self) -> None:
        config = DecisionConfig(ema_alpha=0.3)
        engine = DecisionEngine(config)

        # Feed mostly studying, with one noisy "not studying" frame
        for _ in range(5):
            engine.update(_make_result(studying=0.8, not_studying=0.1))
        engine.update(_make_result(studying=0.1, not_studying=0.8))

        # EMA should still be well above threshold due to smoothing
        assert engine.ema_studying > 0.3


class TestFSM:
    """Test FSM state transitions."""

    def test_starts_studying(self) -> None:
        engine = DecisionEngine(DecisionConfig())
        assert engine.state == StudyState.STUDYING

    def test_transitions_to_distracted(self) -> None:
        config = DecisionConfig(ema_alpha=1.0)  # No smoothing for test
        engine = DecisionEngine(config)

        engine.update(_make_result(studying=0.1, not_studying=0.9))
        assert engine.state == StudyState.DISTRACTED

    def test_single_bad_frame_doesnt_alert(self) -> None:
        config = DecisionConfig(ema_alpha=0.3, distraction_timeout=30.0)
        engine = DecisionEngine(config)

        # One bad frame
        engine.update(_make_result(studying=0.1, not_studying=0.9))

        # Should NOT be ALERT_ACTIVE
        assert engine.state != StudyState.ALERT_ACTIVE

    def test_distracted_recovers_with_good_frames(self) -> None:
        config = DecisionConfig(ema_alpha=1.0, recovery_time=1.0)
        engine = DecisionEngine(config)

        # Go distracted
        engine.update(_make_result(studying=0.1, not_studying=0.9))
        assert engine.state == StudyState.DISTRACTED

        # Mock time to simulate recovery_time elapsed
        with patch("studywatchdog.decision.time") as mock_time:
            # First call: _transition_to in the update that made us DISTRACTED
            # already happened. We need monotonic to return a value that makes
            # time_in_state >= recovery_time when we next call update.
            mock_time.monotonic.return_value = engine._state_entered_at + 2.0
            engine.update(_make_result(studying=0.9, not_studying=0.1))
        assert engine.state == StudyState.STUDYING

    def test_alert_triggers_after_timeout(self) -> None:
        config = DecisionConfig(ema_alpha=1.0, distraction_timeout=5.0)
        engine = DecisionEngine(config)

        # Go distracted
        engine.update(_make_result(studying=0.1, not_studying=0.9))
        assert engine.state == StudyState.DISTRACTED

        # Mock time to simulate distraction_timeout elapsed
        with patch("studywatchdog.decision.time") as mock_time:
            mock_time.monotonic.return_value = engine._state_entered_at + 6.0
            engine.update(_make_result(studying=0.1, not_studying=0.9))
        assert engine.state == StudyState.ALERT_ACTIVE

    def test_alert_stops_when_studying_resumes(self) -> None:
        config = DecisionConfig(ema_alpha=1.0, distraction_timeout=5.0)
        engine = DecisionEngine(config)

        # Go distracted
        engine.update(_make_result(studying=0.1, not_studying=0.9))
        assert engine.state == StudyState.DISTRACTED

        # Mock time to trigger alert
        with patch("studywatchdog.decision.time") as mock_time:
            mock_time.monotonic.return_value = engine._state_entered_at + 6.0
            engine.update(_make_result(studying=0.1, not_studying=0.9))
        assert engine.state == StudyState.ALERT_ACTIVE

        # Resume studying — should stop alert immediately
        engine.update(_make_result(studying=0.9, not_studying=0.1))
        assert engine.state == StudyState.STUDYING

    def test_absent_treated_as_not_studying(self) -> None:
        config = DecisionConfig(ema_alpha=1.0)
        engine = DecisionEngine(config)

        result = DetectionResult(
            status=ActivityStatus.ABSENT,
            confidence=0.9,
            studying_score=0.05,
            not_studying_score=0.05,
            absent_score=0.9,
        )
        engine.update(result)
        assert engine.state == StudyState.DISTRACTED

    def test_reset(self) -> None:
        config = DecisionConfig(ema_alpha=1.0)
        engine = DecisionEngine(config)

        engine.update(_make_result(studying=0.1, not_studying=0.9))
        assert engine.state == StudyState.DISTRACTED

        engine.reset()
        assert engine.state == StudyState.STUDYING
        assert engine.ema_studying == 1.0
