# 🐕 StudyWatchdog

> A local AI assistant that watches you while you study — and if you get distracted for too long... it rickrolls you. 🎵

**StudyWatchdog** uses your webcam and a local vision model (**SigLIP**) to classify in real-time whether you're studying or not. If you stop for too long, it rickrolls you until you resume.

## 🎯 Goal
A fun/educational project to explore local vision AI, not a commercial product.

## 🖥️ Hardware Target
- GPU: NVIDIA RTX A2000 8GB (Laptop)
- CPU: Intel i7-12850HX
- RAM: 32GB
- **Everything runs locally** — no cloud APIs

## 🏗️ Architecture

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

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Start
uv run studywatchdog

# Test
uv run pytest

# Lint & Format
uv run ruff check src/
uv run ruff format src/
```

## 📦 Tech Stack
- **Python 3.12+**
- **uv** — package manager
- **SigLIP** (`google/siglip-base-patch16-224`) — zero-shot image classification
- **OpenCV** — webcam capture
- **PyTorch + Transformers** — model runtime
- **pygame** — audio playback (rickroll!)
- **Ruff** — linting/formatting
- **pytest** — testing

## 🗺️ Roadmap

### Phase 1: Foundation ✅
- [x] Project structure and config
- [x] CLI entry point
- [x] Camera capture working (live preview)

### Phase 2: Detection ✅
- [x] SigLIP zero-shot classification integration
- [x] Decision engine with EMA + FSM
- [ ] Tuning text prompts and thresholds
- [ ] Performance benchmarking on target hardware

### Phase 3: Rickroll 🎵
- [ ] Download/include rickroll audio
- [x] Play/stop controlled by decision engine
- [x] Cooldown and anti-spam

### Phase 4: Polish ✨
- [ ] Data recording for calibration (user as test person)
- [ ] Session statistics (% study time)
- [ ] Alert escalation (nudge → rickroll → TTS roast)
- [ ] System tray / mini GUI

## 📄 License
MIT
