# Copilot Instructions for StudyWatchdog

## ⚠️ CRITICAL: Keep Docs In Sync
Whenever you make changes to architecture, dependencies, modules, config, or design decisions,
you MUST also update: `README.md`, `AGENTS.md`, and this file (`.github/copilot-instructions.md`).
No exceptions.

## Project Overview
StudyWatchdog is a Python application that uses a webcam and **SigLIP** (zero-shot image classification) to detect whether the user is studying or not. If the user stops studying for a configurable period, the system plays a rickroll until they resume. Everything runs locally.

## Detection Approach
- **Model**: SigLIP (`google/siglip-base-patch16-224`) — zero-shot image classification
- **NOT** a VLM/LLM — we do NOT generate text or parse natural language responses
- **Output**: Numerical similarity scores (0.0-1.0) per text candidate
- **Decision**: EMA smoothing + Finite State Machine with temporal tolerance
- **Alert**: Rickroll (interruptible audio playback)

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: `uv` (NOT pip, NOT conda, NOT poetry)
- **AI Model**: SigLIP via `transformers` + `torch`
- **Webcam**: OpenCV (`opencv-python`)
- **Audio**: `pygame` (for rickroll playback)
- **Config**: Pydantic models
- **Linter/Formatter**: Ruff
- **Testing**: pytest
- **Project Layout**: `src/` layout (`src/studywatchdog/`)

## Hardware Target
- GPU: NVIDIA RTX A2000 8GB (Laptop) — CUDA capable
- CPU: Intel i7-12850HX
- RAM: 32GB
- Models must run locally on this hardware

## Architecture
The project is composed of these modules:

| Module | Responsibility |
|---|---|
| `camera.py` | Webcam capture via OpenCV, frame buffering |
| `detector.py` | AI-based study detection (core logic) |
| `alerter.py` | Sound playback, TTS, notification system |
| `config.py` | Settings and configuration management |
| `main.py` | Application entry point, orchestration loop |

## Coding Conventions

### General
- Write all code, comments, and docstrings in **English**
- User-facing messages (logs, alerts, TTS) can be in **Italian**
- Use **type hints** everywhere
- Use **dataclasses** or **Pydantic** for config/data structures
- Prefer **pathlib.Path** over `os.path`
- Use **f-strings** for formatting
- Max line length: **100 characters**
- Follow **PEP 8** via Ruff

### Functions & Classes
- All public functions/methods must have **docstrings** (Google style)
- Keep functions **short and focused** (single responsibility)
- Prefer **composition** over inheritance
- Use **Protocol** classes for abstractions (duck typing)

### Error Handling
- Use specific exceptions, never bare `except:`
- Log errors with context
- Graceful degradation (e.g., if camera fails, explain why)

### Dependencies
- Add dependencies via: `uv add <package>`
- Dev dependencies via: `uv add --dev <package>`
- NEVER use `pip install` directly
- Keep dependencies minimal — every dependency must be justified

### Testing
- Tests go in `tests/`
- Use `pytest` with `uv run pytest`
- Test files named `test_<module>.py`
- Prefer unit tests with mocked I/O (camera, audio)

## AI/ML Specific Guidelines
- Models must fit in **8GB VRAM** (RTX A2000)
- Use **SigLIP** for detection — NOT VLMs, NOT LLMs
- Text candidates (prompts) are **configurable** in config, not hardcoded
- Pre-compute text embeddings at model load time (they don't change per frame)
- Use **lazy loading** for models (don't load until needed)
- Frame capture interval should be **configurable** (default: every 3s)
- Detection should be **async-friendly** (don't block the main loop)
- Cache model instances — never reload per-frame
- Use EMA smoothing + FSM for decision — never trigger on a single frame

## File Organization
```
StudyWatchdog/
├── src/
│   └── studywatchdog/
│       ├── __init__.py
│       ├── main.py          # Entry point & orchestration
│       ├── camera.py         # Webcam capture
│       ├── detector.py       # Study detection AI
│       ├── alerter.py        # Alert system
│       └── config.py         # Configuration
├── tests/
│   ├── __init__.py
│   └── test_detector.py
├── pyproject.toml
└── README.md
```

## Common Commands
```bash
uv run studywatchdog          # Run the application
uv run pytest                 # Run tests
uv run ruff check src/        # Lint
uv run ruff format src/       # Format
uv add <package>              # Add dependency
uv add --dev <package>        # Add dev dependency
```

## Important Notes
- This is a **fun/learning project**, not production software
- Prioritize **simplicity and speed of iteration** over perfection
- It's OK to start with a naive approach and improve iteratively
- Keep the feedback loop short: get something visible working ASAP
- The main detection loop captures a frame → runs detection → decides → alerts
- The rickroll must be **interruptible** — stop playback as soon as studying resumes
- Decision engine uses EMA + FSM — never alert on a single frame
