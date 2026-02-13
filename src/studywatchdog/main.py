"""Main entry point for StudyWatchdog.

Orchestrates the camera → detector → decision engine → alerter pipeline.
Includes debug mode with live camera overlay showing detection results,
interactive toolbar with tooltips, camera switching, and pause/resume.
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from studywatchdog.alerter import Alerter
from studywatchdog.camera import Camera, list_cameras
from studywatchdog.config import AppConfig, CameraConfig, generate_default_config, load_config
from studywatchdog.decision import DecisionEngine, StudyState
from studywatchdog.detector import DetectionResult, SigLIPDetector

logger = logging.getLogger("studywatchdog")

# ── Colors (BGR) ──
C_GREEN = (0, 200, 0)
C_RED = (0, 0, 220)
C_YELLOW = (0, 200, 220)
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_DARK = (30, 30, 30)
C_GRAY = (80, 80, 80)
C_LIGHT_GRAY = (180, 180, 180)
C_BLUE = (200, 120, 40)
C_ORANGE = (0, 140, 255)

STATE_COLORS = {
    StudyState.STUDYING: C_GREEN,
    StudyState.DISTRACTED: C_YELLOW,
    StudyState.ALERT_ACTIVE: C_RED,
}

STATE_LABELS_IT = {
    StudyState.STUDYING: "STAI STUDIANDO",
    StudyState.DISTRACTED: "DISTRATTO...",
    StudyState.ALERT_ACTIVE: "RICKROLL!",
}

STATE_ICONS = {
    StudyState.STUDYING: "OK",
    StudyState.DISTRACTED: "!?",
    StudyState.ALERT_ACTIVE: "!!",
}

# ── Toolbar ──
TOOLBAR_H = 44
BTN_W = 44
BTN_MARGIN = 4


class ToolbarButton:
    """A clickable button in the toolbar."""

    def __init__(
        self,
        key: str,
        icon: str,
        tooltip: str,
        *,
        toggle: bool = False,
    ) -> None:
        self.key = key
        self.icon = icon
        self.tooltip = tooltip
        self.toggle = toggle
        self.active = False
        self.x = 0
        self.y = 0
        self.w = BTN_W
        self.h = TOOLBAR_H - 2 * BTN_MARGIN

    def contains(self, mx: int, my: int) -> bool:
        """Check if a mouse position is inside this button."""
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, frame: np.ndarray, hover: bool = False) -> None:
        """Draw the button on the frame."""
        bg = C_BLUE if self.active else (C_GRAY if hover else C_DARK)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            C_LIGHT_GRAY if hover else C_GRAY,
            1,
        )
        # Center icon text
        (tw, th), _ = cv2.getTextSize(self.icon, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(
            frame, self.icon, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA,
        )


class DebugUI:
    """Debug overlay UI with toolbar, detection info, and interactive controls."""

    WINDOW_NAME = "StudyWatchdog"

    def __init__(self, available_cameras: list[int], current_camera: int) -> None:
        self._mouse_x = 0
        self._mouse_y = 0
        self._available_cameras = available_cameras
        self._current_camera_idx = (
            available_cameras.index(current_camera)
            if current_camera in available_cameras
            else 0
        )
        self._show_scores = True
        self._paused = False

        # Build toolbar buttons
        self._btn_pause = ToolbarButton("pause", "||", "Pausa/Riprendi detection (P)", toggle=True)
        self._btn_cam = ToolbarButton("cam", "CAM", "Cambia telecamera (C)")
        self._btn_scores = ToolbarButton(
            "scores", "SCR", "Mostra/nascondi score (S)", toggle=True
        )
        self._btn_scores.active = True
        self._btn_reset = ToolbarButton("reset", "RST", "Reset stato (R)")
        self._btn_quit = ToolbarButton("quit", "Q", "Esci (Q)")

        self._buttons = [
            self._btn_pause,
            self._btn_cam,
            self._btn_scores,
            self._btn_reset,
            self._btn_quit,
        ]

        # Pending actions from clicks
        self.action_quit = False
        self.action_reset = False
        self.action_switch_camera: int | None = None

    @property
    def paused(self) -> bool:
        """Whether detection is paused."""
        return self._paused

    def setup_window(self) -> None:
        """Create a resizable OpenCV window and set up mouse callback."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.WINDOW_NAME, 800, 600)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        """Handle mouse events."""
        self._mouse_x = x
        self._mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in self._buttons:
                if btn.contains(x, y):
                    self._handle_button_click(btn)
                    break

    def _handle_button_click(self, btn: ToolbarButton) -> None:
        """Process a toolbar button click."""
        if btn.key == "pause":
            self._paused = not self._paused
            btn.active = self._paused
            logger.info("Detection %s", "PAUSED" if self._paused else "RESUMED")
        elif btn.key == "cam":
            self._cycle_camera()
        elif btn.key == "scores":
            self._show_scores = not self._show_scores
            btn.active = self._show_scores
        elif btn.key == "reset":
            self.action_reset = True
        elif btn.key == "quit":
            self.action_quit = True

    def _cycle_camera(self) -> None:
        """Switch to the next available camera."""
        if len(self._available_cameras) <= 1:
            logger.info("Solo una telecamera disponibile")
            return
        self._current_camera_idx = (self._current_camera_idx + 1) % len(
            self._available_cameras
        )
        new_cam = self._available_cameras[self._current_camera_idx]
        self.action_switch_camera = new_cam
        logger.info("Cambio telecamera -> %d", new_cam)

    def handle_key(self, key: int) -> None:
        """Handle keyboard shortcuts."""
        if key == ord("q"):
            self.action_quit = True
        elif key == ord("p"):
            self._handle_button_click(self._btn_pause)
        elif key == ord("c"):
            self._cycle_camera()
        elif key == ord("s"):
            self._handle_button_click(self._btn_scores)
        elif key == ord("r"):
            self.action_reset = True

    def draw(
        self,
        frame: np.ndarray,
        engine: DecisionEngine,
        result: DetectionResult | None,
        fps: float,
        camera_idx: int,
    ) -> np.ndarray:
        """Draw the full debug overlay on the frame.

        Args:
            frame: BGR camera frame.
            engine: Decision engine with current state.
            result: Latest detection result (or None).
            fps: Current FPS.
            camera_idx: Active camera index.

        Returns:
            Frame with overlay drawn (includes toolbar).
        """
        h, w = frame.shape[:2]
        canvas = np.zeros((h + TOOLBAR_H, w, 3), dtype=np.uint8)
        overlay = frame.copy()

        state = engine.state
        color = STATE_COLORS[state]
        label = STATE_LABELS_IT[state]
        icon = STATE_ICONS[state]

        # ── Semi-transparent state banner ──
        banner_h = 48
        sub = overlay[0:banner_h, 0:w]
        banner_bg = np.full_like(sub, color, dtype=np.uint8)
        cv2.addWeighted(banner_bg, 0.75, sub, 0.25, 0, sub)

        cv2.putText(
            overlay, f" {icon} {label}", (8, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_WHITE, 2, cv2.LINE_AA,
        )

        # Time in state (right side)
        tis = engine.time_in_state
        if state == StudyState.DISTRACTED:
            remaining = max(0, engine._config.distraction_timeout - tis)
            time_str = f"alert in {remaining:.0f}s"
        elif state == StudyState.ALERT_ACTIVE:
            time_str = f"rickroll da {tis:.0f}s"
        else:
            time_str = f"{tis:.0f}s"
        cv2.putText(
            overlay, time_str, (w - 180, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA,
        )

        # ── EMA bar ──
        bar_y = banner_h + 6
        bar_h = 22
        bar_margin = 12
        bar_w = w - 2 * bar_margin
        ema = engine.ema_studying
        thresh = engine._config.studying_threshold

        cv2.rectangle(
            overlay, (bar_margin, bar_y), (bar_margin + bar_w, bar_y + bar_h), C_DARK, -1,
        )
        fill_w = max(1, int(bar_w * ema))
        bar_color = C_GREEN if ema >= thresh else C_RED
        cv2.rectangle(
            overlay, (bar_margin, bar_y), (bar_margin + fill_w, bar_y + bar_h), bar_color, -1,
        )
        thresh_x = bar_margin + int(bar_w * thresh)
        cv2.line(overlay, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), C_WHITE, 2)
        cv2.putText(
            overlay, f"EMA: {ema:.2f}", (bar_margin + 4, bar_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA,
        )

        # ── Paused overlay ──
        if self._paused:
            pause_label = "PAUSA"
            (pw, ph), _ = cv2.getTextSize(pause_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            px, py = (w - pw) // 2, h // 2 + ph // 2
            cv2.rectangle(overlay, (px - 16, py - ph - 12), (px + pw + 16, py + 12), C_DARK, -1)
            cv2.putText(
                overlay, pause_label, (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_ORANGE, 3, cv2.LINE_AA,
            )

        # ── Scores panel (right side) ──
        if self._show_scores and result is not None:
            self._draw_scores_panel(overlay, result, w)

        # ── Info line (bottom-left of video) ──
        info_parts = [f"FPS:{fps:.0f}", f"CAM:{camera_idx}"]
        if result:
            info_parts.append(f"Infer:{result.inference_ms:.0f}ms")
        cv2.putText(
            overlay, "  ".join(info_parts), (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_LIGHT_GRAY, 1, cv2.LINE_AA,
        )

        # Compose canvas
        canvas[0:h, 0:w] = overlay
        self._draw_toolbar(canvas, w, h)
        return canvas

    def _draw_scores_panel(
        self, overlay: np.ndarray, result: DetectionResult, w: int
    ) -> None:
        """Draw per-category score bars on the right side."""
        panel_w = 220
        panel_x = w - panel_w - 8
        panel_y = 82
        categories = [
            ("Studio", result.studying_score, C_GREEN),
            ("Distratto", result.not_studying_score, C_YELLOW),
            ("Assente", result.absent_score, C_GRAY),
        ]
        for i, (lbl, score, clr) in enumerate(categories):
            y = panel_y + i * 28
            cv2.rectangle(overlay, (panel_x, y), (panel_x + panel_w, y + 20), C_DARK, -1)
            fill = max(1, int(panel_w * score))
            cv2.rectangle(overlay, (panel_x, y), (panel_x + fill, y + 20), clr, -1)
            cv2.putText(
                overlay, f"{lbl}: {score:.0%}", (panel_x + 4, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1, cv2.LINE_AA,
            )

    def _draw_toolbar(self, canvas: np.ndarray, w: int, video_h: int) -> None:
        """Draw the interactive toolbar at the bottom."""
        toolbar_y = video_h
        cv2.rectangle(canvas, (0, toolbar_y), (w, toolbar_y + TOOLBAR_H), C_DARK, -1)
        cv2.line(canvas, (0, toolbar_y), (w, toolbar_y), C_GRAY, 1)

        # Position and draw buttons
        x = BTN_MARGIN
        for btn in self._buttons:
            btn.x = x
            btn.y = toolbar_y + BTN_MARGIN
            hover = btn.contains(self._mouse_x, self._mouse_y)
            btn.draw(canvas, hover=hover)
            x += btn.w + BTN_MARGIN

        # Tooltip for hovered button
        for btn in self._buttons:
            if btn.contains(self._mouse_x, self._mouse_y):
                self._draw_tooltip(canvas, btn, toolbar_y)
                break

        # Keyboard shortcuts hint (right side)
        cv2.putText(
            canvas, "P=pausa  C=cam  S=score  R=reset  Q=esci",
            (w - 360, toolbar_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_LIGHT_GRAY, 1, cv2.LINE_AA,
        )

    def _draw_tooltip(
        self, canvas: np.ndarray, btn: ToolbarButton, toolbar_y: int
    ) -> None:
        """Draw a tooltip above the toolbar for the given button."""
        text = btn.tooltip
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx, ty = btn.x, toolbar_y - 8
        pad = 6
        cv2.rectangle(
            canvas, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), C_DARK, -1,
        )
        cv2.rectangle(
            canvas, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), C_LIGHT_GRAY, 1,
        )
        cv2.putText(
            canvas, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA,
        )


# ── CLI ──


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="studywatchdog",
        description="AI-powered study monitor with rickroll alerts",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Show debug window with camera feed and detection overlay",
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera device index (default: 0). Use --list-cameras to see options.",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available cameras and exit",
    )
    parser.add_argument(
        "--generate-config", nargs="?", const="auto", default=None,
        metavar="PATH",
        help=(
            "Generate a default config file with documentation and exit. "
            "Writes to ~/.config/studywatchdog/config.toml by default, "
            "or to the given PATH if provided."
        ),
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=None,
        help="Seconds between detection frames (default: 3.0)",
    )
    parser.add_argument(
        "--timeout", "-t", type=float, default=None,
        help="Seconds of distraction before rickroll (default: 30)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
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


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to config."""
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


def _switch_camera(camera: Camera, new_index: int, config: AppConfig) -> Camera:
    """Switch to a different camera device.

    Args:
        camera: Current camera to close.
        new_index: New camera device index.
        config: App config (will be mutated with new index).

    Returns:
        New Camera instance, opened.
    """
    camera.close()
    config.camera.camera_index = new_index
    new_camera = Camera(
        CameraConfig(
            camera_index=new_index,
            capture_interval=config.camera.capture_interval,
            frame_width=config.camera.frame_width,
            frame_height=config.camera.frame_height,
        )
    )
    new_camera.open()
    return new_camera


def main() -> None:
    """Start the StudyWatchdog application."""
    args = parse_args()

    # Handle --list-cameras
    if args.list_cameras:
        cameras = list_cameras()
        if cameras:
            print("Telecamere disponibili:")
            for idx in cameras:
                print(f"  Camera {idx}")
        else:
            print("Nessuna telecamera trovata.")
        sys.exit(0)

    # Handle --generate-config
    if args.generate_config is not None:
        from pathlib import Path as _Path

        out = None if args.generate_config == "auto" else _Path(args.generate_config)
        written = generate_default_config(out)
        print(f"Config file generated: {written}")
        print("Edit it to customize StudyWatchdog, then run: studywatchdog --debug")
        sys.exit(0)

    # Load config
    config_path = None
    if args.config:
        from pathlib import Path

        config_path = Path(args.config)

    config = load_config(config_path)
    _apply_cli_overrides(config, args)
    setup_logging(config.log_level)

    logger.info("StudyWatchdog starting...")
    logger.info("Debug mode: %s", "ON" if config.debug else "OFF")
    logger.info(
        "Camera: %d | Interval: %.1fs | Timeout: %.0fs",
        config.camera.camera_index,
        config.camera.capture_interval,
        config.decision.distraction_timeout,
    )

    # Discover cameras and validate chosen camera
    available_cameras = list_cameras()
    if not available_cameras:
        available_cameras = [config.camera.camera_index]
    logger.info("Telecamere disponibili: %s", available_cameras)

    # If chosen camera isn't available, fall back to first available
    if args.camera is None and config.camera.camera_index not in available_cameras:
        fallback = available_cameras[0]
        logger.warning(
            "Camera %d non disponibile, uso camera %d",
            config.camera.camera_index,
            fallback,
        )
        config.camera.camera_index = fallback

    # Initialize components
    camera = Camera(config.camera)
    detector = SigLIPDetector(config.detector)
    engine = DecisionEngine(config.decision)
    alerter = Alerter(config.alert)

    # Pre-load model
    logger.info("Loading AI model (this may take a moment on first run)...")
    detector.load()
    logger.info("Model ready!")

    # UI
    ui: DebugUI | None = None
    if config.debug:
        ui = DebugUI(available_cameras, config.camera.camera_index)

    # Main loop state
    last_result: DetectionResult | None = None
    frame_count = 0
    fps_start = time.monotonic()
    fps = 0.0

    try:
        camera.open()
        if ui:
            ui.setup_window()
        logger.info("Main loop started. Press Ctrl+C (or Q) to stop.")

        while True:
            frame = camera.read_frame()
            if frame is None:
                logger.error("Lost camera feed. Exiting.")
                break

            # Run detection at configured interval (unless paused)
            if not (ui and ui.paused) and camera.should_capture():
                last_result = detector.detect(frame)
                state = engine.update(last_result)

                if state == StudyState.ALERT_ACTIVE:
                    alerter.play()
                elif state == StudyState.STUDYING:
                    alerter.stop()

            # FPS
            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

            # Debug window
            if ui:
                display = ui.draw(
                    frame, engine, last_result, fps, config.camera.camera_index
                )
                cv2.imshow(ui.WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    ui.handle_key(key)

                # Process pending actions
                if ui.action_quit:
                    logger.info("Quit requested.")
                    break

                if ui.action_reset:
                    engine.reset()
                    alerter.stop()
                    last_result = None
                    logger.info("Manual reset triggered.")
                    ui.action_reset = False

                if ui.action_switch_camera is not None:
                    new_idx = ui.action_switch_camera
                    ui.action_switch_camera = None
                    try:
                        camera = _switch_camera(camera, new_idx, config)
                        engine.reset()
                        alerter.stop()
                        last_result = None
                        logger.info("Switched to camera %d", new_idx)
                    except RuntimeError:
                        logger.error("Cannot open camera %d", new_idx)
            else:
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
