# AGENTS.md — Agent Guidelines for StudyWatchdog

This file provides guidelines for AI coding agents working on this project.

## ⚠️ CRITICAL RULE: Keep Documentation In Sync

**Every time you make a change that affects architecture, dependencies, configuration,
or design decisions, you MUST update the relevant documentation files:**

- **`README.md`** — If the change affects user-facing behavior, features, roadmap, or tech stack
- **`AGENTS.md`** (this file) — If the change affects architecture, coding rules, or development guidelines
- **`.github/copilot-instructions.md`** — If the change affects coding conventions, module responsibilities, or commands

**Examples of changes that require doc updates:**
- Adding/removing a dependency → update tech stack in README + key dependencies in AGENTS.md
- Changing detection approach → update architecture sections everywhere
- Adding a new module → update file organization in copilot-instructions.md
- Changing a config parameter → update config section in AGENTS.md
- Completing a roadmap item → check the checkbox in README.md

**No excuses. If you change the code, check if the docs need updating.**

## 🎯 Project Goal
Build a webcam-based AI monitor that detects if the user is studying and rickrolls them when they stop for too long. The system runs entirely locally on consumer hardware (NVIDIA RTX A2000 8GB, 32GB RAM).

## 🧱 Architecture Overview

```
┌───────────┐    ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐
│  Camera   │───▶│  Detector    │───▶│ Decision Engine │───▶│   Alerter     │
│ (OpenCV)  │    │  (SigLIP)    │    │ (EMA + FSM)     │    │ (Rickroll 🎵) │
└───────────┘    └──────────────┘    └─────────────────┘    └───────────────┘
                       │                      │
                       │      ┌────────┐      │
                       └─────▶│ Config │◀─────┘
                              └────────┘
```

### Data Flow
1. **Camera** captures a frame every N seconds (configurable, default 3s)
2. **Detector (SigLIP)** performs zero-shot image classification against text candidates, returns numerical scores (0.0-1.0)
3. **Decision Engine** applies EMA smoothing on scores, then a Finite State Machine manages state transitions with temporal tolerance
4. **Alerter** plays rickroll when distraction timeout is exceeded, stops it when studying resumes

### Why SigLIP (not a VLM/LLM)

We use **SigLIP** (`google/siglip-base-patch16-224`) for zero-shot image classification because:

- **Numerical output**: Returns similarity scores (0.0-1.0), no text parsing needed
- **Fast**: ~20-50ms per frame on GPU (vs 1-3s for VLMs)
- **Small**: ~0.2B params, ~400MB, ~1GB VRAM
- **Deterministic**: Same input → same output (no randomness in generation)
- **Robust**: No hallucinations, no parsing failures, no prompt injection
- **Tunable**: Adjust text candidates to improve classification without retraining

**How it works**: We define text candidates like "a person studying at a desk" and "a person distracted, looking at their phone". SigLIP computes cosine similarity between the image embedding and each text embedding, then applies sigmoid to get per-class probabilities. The highest score wins.

### Decision Engine

The decision engine prevents single-frame noise from triggering false alerts:

1. **EMA (Exponential Moving Average)**: Smooths raw detector scores over time
   - `ema_t = alpha * score_t + (1 - alpha) * ema_{t-1}`
   - `alpha = 0.3` (configurable) — higher = more reactive, lower = more smooth

2. **FSM (Finite State Machine)** with states:
   - `STUDYING` → transition to `DISTRACTED` when EMA drops below threshold for N seconds
   - `DISTRACTED` → transition to `STUDYING` when EMA rises above threshold for M seconds
   - `DISTRACTED` → transition to `ALERT_ACTIVE` after distraction timeout
   - `ALERT_ACTIVE` → transition to `STUDYING` when EMA rises above threshold (stops rickroll)

### Alert System (Rickroll)

- **Primary alert**: Play "Never Gonna Give You Up" (rickroll)
- **Interruptible**: Stops immediately when the user resumes studying
- **Cooldown**: Configurable minimum interval between alerts (anti-spam)
- **Future escalation**: gentle nudge → rickroll → TTS roast

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
- ✅ **Update README.md, AGENTS.md, and copilot-instructions.md when making architectural changes**

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
- ❌ Use a VLM/LLM for detection — use SigLIP zero-shot classification
- ❌ Parse natural language model output for classification — use numerical scores
- ❌ **Make changes without checking if docs need updating**

### When Implementing the Detector
1. Use **SigLIP** (`google/siglip-base-patch16-224`) for zero-shot classification
2. The detector interface should be a **Protocol** so implementations are swappable
3. Every detector must implement: `detect(frame: np.ndarray) -> DetectionResult`
4. `DetectionResult` should include: `status` (enum), `confidence` (float), `scores` (dict[str, float])
5. Text candidates are **configurable** — they live in config, not hardcoded
6. Log inference time for performance monitoring
7. Pre-compute text embeddings at startup (they don't change per frame)

### When Implementing the Decision Engine
1. Use **EMA** to smooth per-frame scores: `ema = alpha * score + (1 - alpha) * prev_ema`
2. Use a **FSM** (Finite State Machine) for state transitions
3. States: `STUDYING`, `DISTRACTED`, `ALERT_ACTIVE`
4. All thresholds and timeouts must be in **config**
5. A single "distracted" frame MUST NOT trigger an alert
6. Log state transitions for debugging

### When Implementing Alerts
1. Use `pygame.mixer` for rickroll audio playback
2. Audio must be **interruptible** — stop() when studying resumes
3. Alerts should have configurable **cooldown** (don't spam the user)
4. Future: escalation system (gentle nudge → rickroll → TTS roast)
5. The rickroll audio file should be in an `assets/` directory

### When Modifying Config
1. Use Pydantic models for validation
2. Support loading from TOML file (same format as pyproject.toml)
3. All magic numbers must be in config, not scattered in code
4. Provide sensible defaults for everything

## 🧪 Testing Strategy
- **Unit tests**: Test detector logic with sample images, test decision engine state transitions, test alerter with mocked audio
- **Integration tests**: Test the main loop with a mocked camera
- **Manual testing**: Provide a CLI flag to use a video file instead of live camera
- **Calibration data**: The user can record themselves studying/not-studying for tuning

## 📦 Key Dependencies (expected)
| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture and image processing |
| `torch` | ML model runtime (CUDA) |
| `transformers` | SigLIP model loading and inference |
| `Pillow` | Image conversion (OpenCV → PIL for SigLIP) |
| `pydantic` | Configuration models with validation |
| `pygame` | Audio playback (rickroll) |

## 🗺️ Development Roadmap

### Phase 1: Foundation ✅
- [x] Basic project structure and config
- [x] CLI entry point
- [x] Camera capture working (show live preview)

### Phase 2: Detection ✅
- [x] SigLIP zero-shot classification integration
- [x] Decision engine with EMA + FSM
- [ ] Tuning text candidates and thresholds
- [ ] Performance benchmarking on target hardware

### Phase 3: Rickroll 🎵
- [x] Audio playback with play/stop control
- [x] Rickroll triggered by decision engine
- [x] Cooldown and anti-spam

### Phase 4: Polish ✨
- [ ] Recording mode for calibration data
- [ ] Session statistics (% time studying)
- [ ] Alert escalation system
- [ ] System tray / mini GUI
