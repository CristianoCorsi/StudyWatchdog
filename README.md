# 🐕 StudyWatchdog

> Un assistente AI locale che ti tiene d'occhio mentre studi — e se ti distrai troppo a lungo... parte il rickroll. 🎵

**StudyWatchdog** usa la webcam e un modello di visione locale (**SigLIP**) per classificare in tempo reale se stai studiando o meno. Se smetti per troppo tempo, ti rickrolla finché non riprendi.

## 🎯 Goal
Un progetto fun/didattico per esplorare vision AI locale, non un prodotto commerciale.

## 🖥️ Hardware Target
- GPU: NVIDIA RTX A2000 8GB (Laptop)
- CPU: Intel i7-12850HX
- RAM: 32GB
- **Tutto gira in locale** — nessuna API cloud

## 🏗️ Architecture

```
┌───────────┐    ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐
│  Camera   │───▶│  Detector    │───▶│ Decision Engine  │───▶│   Alerter     │
│ (OpenCV)  │    │  (SigLIP)    │    │ (EMA + FSM)      │    │ (Rickroll 🎵) │
└───────────┘    └──────────────┘    └─────────────────┘    └───────────────┘
                       │                      │
                       │      ┌────────┐      │
                       └─────▶│ Config │◀─────┘
                              └────────┘
```

### Loop Principale
1. **Camera** cattura un frame ogni N secondi (default: 3s)
2. **Detector (SigLIP)** calcola similarità immagine vs testi candidati → score numerico 0.0-1.0
3. **Decision Engine** applica un EMA (Exponential Moving Average) sugli score per smussare i risultati, poi una FSM (Finite State Machine) decide le transizioni di stato
4. **Alerter** avvia il rickroll quando il timeout di distrazione è superato, lo stoppa quando si riprende a studiare

### Perché SigLIP e non un LLM/VLM?

| Criterio | SigLIP (zero-shot classification) | VLM (moondream, LLaVA...) |
|---|---|---|
| **Output** | Score numerico 0.0-1.0, diretto | Testo libero da parsare (fragile!) |
| **Velocità** | ~20-50ms per frame su GPU | ~1-3s per frame |
| **Dimensione** | ~0.2B params, ~400MB | ~2B+ params, ~4GB+ |
| **Determinismo** | Stesso input → stesso output | Può variare ad ogni run |
| **Robustezza** | Nessun parsing, nessuna hallucination | Il modello può "inventare" |
| **Soglie** | Configurabili numericamente | Bisogna interpretare testo |
| **VRAM** | ~1GB | ~3-4GB |

**SigLIP** è un modello contrastivo (come CLIP, ma migliore) che confronta un'immagine con delle descrizioni testuali e restituisce uno **score di similarità numerico** per ciascuna. Nessun testo da generare, nessun parsing, nessuna hallucination — solo numeri.

### Come Funziona la Detection

```python
# Pseudocodice del detector
texts = [
    "a person studying, reading a book, or working focused at a desk",
    "a person distracted, looking at phone, not paying attention",
    "an empty desk, no person visible",
]
scores = siglip(image, texts)  # → [0.82, 0.15, 0.03]
# Il più alto vince → "studying" con confidence 0.82
```

I prompt testuali sono **configurabili**: se la classificazione non è buona su certi edge case, basta modificare le descrizioni testuali senza toccare codice o rifare training.

### Decision Engine: Tolleranza Temporale

Non basta un singolo frame per decidere — il sistema usa:

1. **EMA (Exponential Moving Average)** sugli score di confidence per smussare rumore e flicker:
   - $\text{EMA}_t = \alpha \cdot \text{score}_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$
   - Con $\alpha = 0.3$ (configurabile) → i singoli spike vengono attenuati

2. **FSM (Finite State Machine)** con 3 stati e transizioni a tempo:
   ```
   STUDYING ──(EMA < soglia per N secondi)──▶ DISTRACTED
   DISTRACTED ──(EMA > soglia per M secondi)──▶ STUDYING
   DISTRACTED ──(timeout superato)──▶ ALERT_ACTIVE (rickroll!)
   ALERT_ACTIVE ──(EMA > soglia)──▶ STUDYING (rickroll stop)
   ```

3. **Parametri configurabili**:
   - `distraction_timeout`: secondi prima dell'alert (default: 30s)
   - `recovery_time`: secondi di studio per uscire dallo stato distratto (default: 5s)
   - `studying_threshold`: soglia EMA per "sta studiando" (default: 0.5)
   - `ema_alpha`: peso dell'ultimo frame nell'EMA (default: 0.3)

### 🎵 Il Rickroll

Quando il decision engine decide che sei distratto da troppo tempo:
- Parte **"Never Gonna Give You Up"** di Rick Astley
- La riproduzione è **interrompibile**: appena ricominci a studiare, si stoppa
- Se ti ri-distrai, riparte (con cooldown configurabile per non essere troppo aggressivo)
- In futuro: escalation (prima un nudge gentile, poi il rickroll completo, poi TTS roast)

## 🚀 Quick Start

```bash
# Installa le dipendenze
uv sync

# Avvia
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

### Phase 1: Foundation ✏️
- [x] Struttura progetto e config
- [x] Entry point CLI
- [ ] Camera capture funzionante (preview live)

### Phase 2: Detection 🧠
- [ ] Integrazione SigLIP zero-shot classification
- [ ] Decision engine con EMA + FSM
- [ ] Tuning dei prompt testuali e soglie
- [ ] Benchmark performance su hardware target

### Phase 3: Rickroll 🎵
- [ ] Download/inclusione audio rickroll
- [ ] Play/stop controllato dal decision engine
- [ ] Cooldown e anti-spam

### Phase 4: Polish ✨
- [ ] Registrazione dati per calibrazione (utente come test person)
- [ ] Statistiche sessione (% tempo studio)
- [ ] Alert escalation (nudge → rickroll → TTS roast)
- [ ] System tray / mini GUI

## 📄 License
MIT
