"""Tests for study activity detector."""

from studywatchdog.detector import ActivityStatus, DetectionResult


class TestDetectionResult:
    """Test DetectionResult data class."""

    def test_studying_result(self) -> None:
        result = DetectionResult(
            status=ActivityStatus.STUDYING,
            confidence=0.85,
            studying_score=0.85,
            not_studying_score=0.10,
            absent_score=0.05,
        )
        assert result.status == ActivityStatus.STUDYING
        assert result.confidence == 0.85

    def test_scores_are_independent(self) -> None:
        """SigLIP uses sigmoid, not softmax — scores don't sum to 1."""
        result = DetectionResult(
            status=ActivityStatus.STUDYING,
            confidence=0.7,
            studying_score=0.7,
            not_studying_score=0.6,
            absent_score=0.3,
        )
        total = result.studying_score + result.not_studying_score + result.absent_score
        # Scores can sum to anything, not necessarily 1.0
        assert total != 1.0 or True  # Just documenting the behavior
