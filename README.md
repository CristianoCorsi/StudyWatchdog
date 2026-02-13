# 🐕 StudyWatchdog

> Your AI study companion that keeps you focused — with a twist! Get distracted for too long, and you'll be rickrolled. 🎵

**StudyWatchdog** is a local AI-powered study monitor that uses your webcam and computer vision to detect when you're studying vs. distracted. After a configurable period of distraction, it plays "Never Gonna Give You Up" until you get back to work.

## Features

- **Real-time Study Detection** — Uses SigLIP zero-shot image classification for accurate activity recognition
- **Smart Decision Engine** — EMA smoothing + Finite State Machine prevents false alerts from brief glances away
- **Rickroll Alerts** — Interruptible audio playback that stops when you resume studying
- **100% Local** — No cloud APIs, no data collection, complete privacy
- **Highly Configurable** — Adjust detection sensitivity, timeouts, text prompts, and more
- **Debug Mode** — Live camera overlay with detection scores, state visualization, and controls

## Project Goals

This is a fun/educational project designed to:
- Explore **local vision AI** capabilities on consumer hardware
- Demonstrate **zero-shot image classification** with SigLIP
- Learn about **state machines** and **temporal signal processing** (EMA)
- Build a practical (and entertaining) productivity tool

**Not intended as a commercial product** — use it, learn from it, remix it!

## Requirements

### Hardware
- **Webcam** (built-in or USB)
- **GPU**: NVIDIA GPU with CUDA support recommended (runs on ~1GB VRAM)
  - Also works on CPU, but slower (~300-500ms vs ~20-50ms per frame)
- **RAM**: 4GB+ recommended

### Software
- **Python**: 3.12 or higher
- **OS**: Linux, macOS, or Windows
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (fast Python package installer)

## Architecture

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

### Main Loop
1. **Camera** captures a frame every N seconds (default: 3s)
2. **Detector (SigLIP)** computes image similarity vs text candidates → numerical score 0.0-1.0
3. **Decision Engine** applies EMA (Exponential Moving Average) on scores to smooth results, then an FSM (Finite State Machine) decides state transitions
4. **Alerter** starts the rickroll when distraction timeout is exceeded, stops it when studying resumes

### Why SigLIP and not an LLM/VLM?

| Criterion | SigLIP (zero-shot classification) | VLM (moondream, LLaVA...) |
|---|---|---|
| **Output** | Numerical score 0.0-1.0, direct | Free text to parse (fragile!) |
| **Speed** | ~20-50ms per frame on GPU | ~1-3s per frame |
| **Size** | ~0.2B params, ~400MB | ~2B+ params, ~4GB+ |
| **Determinism** | Same input → same output | Can vary on each run |
| **Robustness** | No parsing, no hallucination | The model can "invent" |
| **Thresholds** | Numerically configurable | Need to interpret text |
| **VRAM** | ~1GB | ~3-4GB |

**SigLIP** is a contrastive model (like CLIP, but better) that compares an image with text descriptions and returns a **numerical similarity score** for each. No text to generate, no parsing, no hallucinations — just numbers.

### How Detection Works

```python
# Detector pseudocode
texts = [
    "a person studying, reading a book, or working focused at a desk",
    "a person distracted, looking at phone, not paying attention",
    "an empty desk, no person visible",
]
scores = siglip(image, texts)  # → [0.82, 0.15, 0.03]
# Highest wins → "studying" with confidence 0.82
```

Text prompts are **configurable**: if classification isn't good on certain edge cases, just modify the text descriptions without touching code or retraining.

### Decision Engine: Temporal Tolerance

A single frame isn't enough to decide — the system uses:

1. **EMA (Exponential Moving Average)** on confidence scores to smooth noise and flicker:
   - $\text{EMA}_t = \alpha \cdot \text{score}_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$
   - With $\alpha = 0.3$ (configurable) → individual spikes are attenuated

2. **FSM (Finite State Machine)** with 3 states and time-based transitions:
   ```
   STUDYING ──(EMA < threshold for N seconds)──▶ DISTRACTED
   DISTRACTED ──(EMA > threshold for M seconds)──▶ STUDYING
   DISTRACTED ──(timeout exceeded)──▶ ALERT_ACTIVE (rickroll!)
   ALERT_ACTIVE ──(EMA > threshold)──▶ STUDYING (rickroll stop)
   ```

3. **Configurable parameters**:
   - `distraction_timeout`: seconds before alert (default: 30s)
   - `recovery_time`: seconds of studying to exit distracted state (default: 5s)
   - `studying_threshold`: EMA threshold for "is studying" (default: 0.5)
   - `ema_alpha`: weight of latest frame in EMA (default: 0.3)

### 🎵 The Rickroll

When the decision engine decides you've been distracted too long:
- Plays **"Never Gonna Give You Up"** by Rick Astley
- Playback is **interruptible**: as soon as you resume studying, it stops
- If you get distracted again, it restarts (with configurable cooldown to avoid being too aggressive)
- Future: escalation (first a gentle nudge, then full rickroll, then TTS roast)

## Quick Start

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/CristianoCorsi/StudyWatchdog.git
   cd StudyWatchdog
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Add rickroll audio** (optional, for alerts):
   - Download `rickroll.mp3` and place it in the `assets/` folder
   - Or use any MP3 file of your choice (configurable)

### Usage

**List available cameras**:
```bash
uv run studywatchdog --list-cameras
```✅ Completed
- [x] Core detection engine with SigLIP
- [x] EMA + FSM decision system
- [x] Rickroll alert system
- [x] Debug overlay UI with live camera feed
- [x] Interactive toolbar and keyboard shortcuts
- [x] Multi-camera support
- [x] Full configuration system

### In Progress
- [ ] Performance benchmarks across different hardware
- [ ] Tuning default text prompts for better accuracy

### Future Ideas
- [ ] Session statistics dashboard (% time studying, focus streaks)
- [ ] Alert escalation system (gentle nudge → rickroll → TTS roast)
- [ ] Recording mode for calibration data
- [ ] System tray / mini GUI
- [ ] Cross-platform packaging (AppImage, .exe, .app)

## Contributing

Contributions are welcome! This is a learning project, so feel free to:
- Report bugs or suggest features via [Issues](https://github.com/CristianoCorsi/StudyWatchdog/issues)
- Submit Pull Requests with improvements
- Share your configuration tweaks or text prompts that work well for your setup

Please ensure:
- Code follows the existing style (use `ruff format`)
- Tests pass (`uv run pytest`)
- New features include tests where applicable

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **SigLIP** by Google Research — [paper](https://arxiv.org/abs/2303.15343)
- **Transformers** by HuggingFace — [library](https://github.com/huggingface/transformers)
- **uv** by Astral — [package manager](https://github.com/astral-sh/uv)

---

**Disclaimer**: This project is for educational and entertainment purposes. Use responsibly and ensure you comply with local privacy laws when using webcam monitoring software.C` | Cycle through cameras |
| `S` | Show/hide score panel |
| `R` | Reset state |
| `Q` | Quit |

## Configuration

All parameters are configurable via TOML file or CLI arguments:

```bash
# Override camera index
uv run studywatchdog --camera 0

# Adjust detection interval
uv run studywatchdog --interval 5.0

# Change distraction timeout
uv run studywatchdog --timeout 60
```

See `uv run studywatchdog --generate-config` for a fully documented config file with all available options.

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=studywatchdog

# Lint & Format
uv run ruff check src/
uv run ruff format src/
```

## Tech Stack
- **Python 3.12+**
- **uv** — package manager
- **SigLIP** (`google/siglip-base-patch16-224`) — zero-shot image classification
- **OpenCV** — webcam capture
- **PyTorch + Transformers** — model runtime
- **pygame** — audio playback (rickroll!)
- **Ruff** — linting/formatting
- **pytest** — testing

## Roadmap

### Completed
- [x] Core detection engine with SigLIP
- [x] EMA + FSM decision system
- [x] Rickroll alert system
- [x] Debug overlay UI with live camera feed
- [x] Interactive toolbar and keyboard shortcuts
- [x] Multi-camera support
- [x] Full configuration system

### In Progress
- [ ] Performance benchmarks across different hardware
- [ ] Tuning default text prompts for better accuracy

### Future Ideas
- [ ] Session statistics dashboard (% time studying, focus streaks)
- [ ] Alert escalation system (gentle nudge → rickroll → TTS roast)
- [ ] Recording mode for calibration data
- [ ] System tray / mini GUI
- [ ] Cross-platform packaging (AppImage, .exe, .app)

## Contributing

Contributions are welcome! This is a learning project, so feel free to:
- Report bugs or suggest features via [Issues](https://github.com/CristianoCorsi/StudyWatchdog/issues)
- Submit Pull Requests with improvements
- Share your configuration tweaks or text prompts that work well for your setup

Please ensure:
- Code follows the existing style (use `ruff format`)
- Tests pass (`uv run pytest`)
- New features include tests where applicable

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments
