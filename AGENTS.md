# AGENTS.md — Agent Guidelines for StudyWatchdog

This file provides guidelines for AI coding agents working on this project.

## 🎯 Project Goal
Build a webcam-based AI monitor that detects if the user is studying and alerts them when they stop for too long. The system runs entirely locally on consumer hardware (NVIDIA RTX A2000 8GB, 32GB RAM).

## 🧱 Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌───────────┐
│   Camera     │────▶│   Detector    │────▶│  Decision   │────▶│  Alerter   │
│  (OpenCV)    │     │  (AI Model)   │     │   Engine    │     │ (Sound/TTS)│
└─────────────┘     └──────────────┘     └────────────┘     └───────────┘
      │                                         │
      │              ┌──────────────┐           │
      └─────────────▶│    Config     │◀──────────┘
                     └──────────────┘
```

### Data Flow
1. **Camera** captures a frame every N seconds (configurable, default 5s)
2. **Detector** analyzes the frame and returns a classification: `studying` | `not_studying` | `absent`
3. **Decision Engine** (in main loop) tracks state over time — a single frame of "not studying" is not enough to trigger an alert
4. **Alerter** fires when the distraction timeout is exceeded

## 📏 Rules for Agents

### DO
- ✅ Use `uv` for ALL package management (`uv add`, `uv run`)
- ✅ Write type hints on all function signatures
- ✅ Write docstrings (Google style) on all public functions
- ✅ Keep modules focused — one responsibility per file
- ✅ Test with `pytest` (mock camera/audio in tests)
- ✅ Use Ruff for linting and formatting
- ✅ Prefer standard library when possible
- ✅ Use `logging` module (not `print()`) for debug/info output
- ✅ Handle errors gracefully with informative messages
- ✅ Consider GPU memory constraints (8GB VRAM total)

### DON'T
- ❌ Use `pip install` — always use `uv add`
- ❌ Use cloud APIs — everything must run locally
- ❌ Add unnecessary dependencies
- ❌ Write overly complex abstractions for a fun project
- ❌ Load AI models eagerly at import time
- ❌ Block the main thread with synchronous model inference without a timeout
- ❌ Hardcode paths, thresholds, or model names — use config
- ❌ Ignore the `src/` layout — all source code lives in `src/studywatchdog/`
- ❌ Create new top-level Python files — use the module structure

### When Implementing the Detector
1. Start with the simplest approach that works
2. The detector interface should be a **Protocol** so implementations are swappable
3. Every detector must implement: `detect(frame: np.ndarray) -> DetectionResult`
4. `DetectionResult` should include: `status` (enum), `confidence` (float), `details` (dict)
5. Log inference time for performance monitoring

### When Implementing Alerts
1. Start with simple sound playback (e.g., `playsound` or `pygame.mixer`)
2. TTS can be added later (e.g., `pyttsx3` or `edge-tts`)
3. Alerts should have configurable cooldown (don't spam the user)
4. Escalation is a nice-to-have: gentle nudge → louder alert → TTS roast

### When Modifying Config
1. Use a dataclass or Pydantic model
2. Support loading from a YAML/TOML file
3. All magic numbers must be in config, not scattered in code
4. Provide sensible defaults

## 🧪 Testing Strategy
- **Unit tests**: Test detector logic with sample images, test alerter with mocked audio
- **Integration tests**: Test the main loop with a mocked camera
- **Manual testing**: Provide a CLI flag to use a video file instead of live camera

## 📦 Key Dependencies (expected)
| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture and image processing |
| `torch` | ML model runtime |
| `transformers` | Hugging Face model loading |
| `Pillow` | Image conversion |
| `pydantic` | Configuration models |

Additional dependencies TBD based on chosen detection approach.

## 🗺️ Development Roadmap

### Phase 1: Foundation ✏️
- [ ] Camera capture working (show live preview)
- [ ] Basic project structure and config
- [ ] CLI entry point

### Phase 2: Detection 🧠
- [ ] Choose and integrate detection model
- [ ] Implement detector with Protocol interface
- [ ] Basic "studying vs not" classification
- [ ] Performance benchmarking on target hardware

### Phase 3: Alerts 🔔
- [ ] Sound playback on distraction
- [ ] Configurable timeout and cooldown
- [ ] Basic TTS integration

### Phase 4: Polish ✨
- [ ] Simple GUI or system tray integration
- [ ] Session statistics (% time studying)
- [ ] Fine-tune detection accuracy
- [ ] Alert escalation system
