"""Main entry point for StudyWatchdog.

Orchestrates the camera → detector → decision engine → alerter pipeline.
Includes debug mode with live camera overlay showing detection results.
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from studywatchdog.alerter import Alerter
from studywatchdog.camera import Camera, list_cameras
from studywatchdog.config import load_config
from studywatchdog.decision import DecisionEngine, StudyState
from studywatchdog.detector import DetectionResult, SigLIPDetector

logger = logging.getLogger("studywatchdog")

# Colors for debug overlay (BGR)
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 220)
COLOR_YELLOW = (0, 200, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BG_DARK = (30, 30, 30)

STATE_COLORS = {
    StudyState.STUDYING: COLOR_GREEN,
    StudyState.DISTRACTED: COLOR_YELLOW,
    StudyState.ALERT_ACTIVE: COLOR_RED,
}

STATE_LABELS = {
    StudyState.STUDYING: "STUDYING",
    StudyState.DISTRACTED: "DISTRACTED",
    StudyState.ALERT_ACTIVE: "RICKROLL!",
}


def draw_debug_overlay(
    frame: np.ndarray,
    engine: DecisionEngine,
    result: DetectionResult | None,
    fps: float,
) -> np.ndarray:
    """Draw debug information overlay on the camera frame.

    Shows: FSM state, EMA score, per-category scores, inference time, FPS.

    Args:
        frame: BGR camera frame.
        engine: Decision engine with current state.
        result: Latest detection result (or None if not yet available).
        fps: Current main loop FPS.

    Returns:
        Frame with overlay drawn.
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # ── State banner at top ──
    state = engine.state
    color = STATE_COLORS[state]
    label = STATE_LABELS[state]
    banner_h = 50
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.putText(
        overlay,
        label,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        COLOR_BLACK,
        2,
        cv2.LINE_AA,
    )
    time_str = f"in state: {engine.time_in_state:.0f}s"
    cv2.putText(
        overlay,
        time_str,
        (w - 200, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_BLACK,
        1,
        cv2.LINE_AA,
    )

    # ── EMA bar ──
    bar_y = banner_h + 10
    bar_h = 25
    bar_w = w - 20
    ema = engine.ema_studying

    # Background
    cv2.rectangle(overlay, (10, bar_y), (10 + bar_w, bar_y + bar_h), COLOR_BG_DARK, -1)
    # Filled portion
    fill_w = int(bar_w * ema)
    bar_color = COLOR_GREEN if ema >= 0.5 else COLOR_RED
    cv2.rectangle(overlay, (10, bar_y), (10 + fill_w, bar_y + bar_h), bar_color, -1)
    # Label
    cv2.putText(
        overlay,
        f"EMA: {ema:.2f}",
        (15, bar_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    # Threshold marker
    thresh_x = int(10 + bar_w * 0.5)
    cv2.line(overlay, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), COLOR_WHITE, 2)

    # ── Detection scores panel ──
    panel_y = bar_y + bar_h + 10
    if result is not None:
        lines = [
            f"Study:    {result.studying_score:.2f}",
            f"Distract: {result.not_studying_score:.2f}",
            f"Absent:   {result.absent_score:.2f}",
            f"Infer:    {result.inference_ms:.0f}ms",
        ]
        for i, line in enumerate(lines):
            y = panel_y + 20 + i * 22
            cv2.putText(
                overlay,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )
        panel_y = y + 10

    # ── FPS counter bottom-right ──
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(
        overlay,
        fps_text,
        (w - 100, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    # ── Key hints bottom-left ──
    cv2.putText(
        overlay,
        "Q=quit  D=toggle debug  R=reset",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    return overlay


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="studywatchdog",
        description="AI-powered study monitor with rickroll alerts",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show debug window with camera feed and detection overlay",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="Camera device index (default: 0). Use --list-cameras to see options.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=None,
        help="Seconds between detection frames (default: 3.0)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=None,
        help="Seconds of distraction before rickroll (default: 30)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Start the StudyWatchdog application."""
    args = parse_args()

    # Handle --list-cameras
    if args.list_cameras:
        cameras = list_cameras()
        if cameras:
            print("Available cameras:")
            for idx in cameras:
                print(f"  Camera {idx}")
        else:
            print("No cameras found.")
        sys.exit(0)

    # Load config
    config_path = None
    if args.config:
        from pathlib import Path

        config_path = Path(args.config)

    config = load_config(config_path)

    # CLI overrides
    if args.debug:
        config.debug = True
    if args.camera is not None:
        config.camera.camera_index = args.camera
    if args.interval is not None:
        config.camera.capture_interval = args.interval
    if args.timeout is not None:
        config.decision.distraction_timeout = args.timeout
    if args.log_level is not None:
        config.log_level = args.log_level

    setup_logging(config.log_level)

    logger.info("StudyWatchdog starting...")
    logger.info("Debug mode: %s", "ON" if config.debug else "OFF")
    logger.info(
        "Camera: %d | Interval: %.1fs | Timeout: %.0fs",
        config.camera.camera_index,
        config.camera.capture_interval,
        config.decision.distraction_timeout,
    )

    # Initialize components
    camera = Camera(config.camera)
    detector = SigLIPDetector(config.detector)
    engine = DecisionEngine(config.decision)
    alerter = Alerter(config.alert)

    # Pre-load model (so the user doesn't wait on first detection)
    logger.info("Loading AI model (this may take a moment on first run)...")
    detector.load()
    logger.info("Model ready!")

    # Main loop
    last_result: DetectionResult | None = None
    frame_count = 0
    fps_start = time.monotonic()
    fps = 0.0
    show_debug = config.debug

    try:
        camera.open()
        logger.info("Main loop started. Press Ctrl+C (or Q in debug window) to stop.")

        while True:
            frame = camera.read_frame()
            if frame is None:
                logger.error("Lost camera feed. Exiting.")
                break

            # Run detection at configured interval
            if camera.should_capture():
                last_result = detector.detect(frame)
                state = engine.update(last_result)

                # Act on state
                if state == StudyState.ALERT_ACTIVE:
                    alerter.play()
                elif state == StudyState.STUDYING:
                    alerter.stop()

            # FPS calculation
            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            # Debug window
            if show_debug:
                display = draw_debug_overlay(frame, engine, last_result, fps)
                cv2.imshow("StudyWatchdog - Debug", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested via debug window.")
                    break
                elif key == ord("d"):
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyAllWindows()
                elif key == ord("r"):
                    engine.reset()
                    alerter.stop()
                    logger.info("Manual reset triggered.")
            else:
                # Without debug window, we need a small sleep to not peg the CPU
                # but still poll keyboard via OpenCV if it was toggled on before
                time.sleep(0.03)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        alerter.stop()
        alerter.cleanup()
        camera.close()
        cv2.destroyAllWindows()
        logger.info("StudyWatchdog stopped.")


if __name__ == "__main__":
    main()
