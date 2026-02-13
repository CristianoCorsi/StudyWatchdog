# 🐕 StudyWatchdog

> Un assistente AI locale che ti tiene d'occhio mentre studi — e ti richiama se ti distrai troppo.

**StudyWatchdog** usa la webcam e modelli AI locali per capire se stai studiando o meno. Se smetti per troppo tempo, ti avvisa con suoni, TTS, o notifiche.

## 🎯 Goal
Un progetto fun/didattico per esplorare vision AI locale, non un prodotto commerciale.

## 🖥️ Hardware Target
- GPU: NVIDIA RTX A2000 8GB (Laptop)
- CPU: Intel i7-12850HX
- RAM: 32GB
- **Tutto gira in locale** — nessuna API cloud

## 🏗️ Architecture

```
Camera → Detector (AI) → Decision Engine → Alerter
              ↑                    ↑
              └──── Config ────────┘
```

Loop principale:
1. Cattura frame dalla webcam (ogni 5s configurabili)
2. Il detector classifica: `studying` | `not_studying` | `absent`
3. Il decision engine traccia lo stato nel tempo
4. L'alerter si attiva dopo un timeout di distrazione

## 🧠 Opzioni per la Detection

### Opzione A: Small Vision-Language Model (VLM) — ⭐ Consigliata
**Modello: [moondream2](https://github.com/vikhyat/moondream)** (~2B parametri)

- **Pro**: Molto flessibile — basta chiedere "is this person studying?" in linguaggio naturale
- **Pro**: Capisce il contesto visivo (libri, laptop, postura, ecc.)
- **Pro**: Sta comodamente in 8GB VRAM (anche in fp16)
- **Pro**: Può dare risposte articolate, non solo classificazione binaria
- **Con**: Più lento (~1-3s per frame su A2000), ma accettabile per analisi ogni 5s
- **Con**: Può essere inconsistente su edge cases

**Come funziona**: Dai un'immagine al modello + un prompt tipo "Describe what the person is doing. Are they studying or distracted?" e il modello risponde in testo. Si parsa la risposta per determinare lo stato.

### Opzione B: MediaPipe Pose Estimation + Regole
- **Pro**: Ultra-veloce (<50ms), leggero, nessuna GPU necessaria
- **Pro**: Deterministico e prevedibile
- **Con**: Molto limitato — può dire "persona seduta al desk" ma non se sta studiando vs scrollando Instagram
- **Con**: Richiede regole manuali (fragile)

### Opzione C: YOLO Object Detection
- **Pro**: Veloce, affidabile per detection di oggetti (libri, laptop, telefono)
- **Con**: Rileva oggetti, non attività — un libro aperto non vuol dire che stai leggendo

### Opzione D: Classificatore Custom (fine-tuned)
- **Pro**: Potenzialmente molto accurato
- **Con**: Richiede raccolta dati e training — troppo effort per un progetto fun

### 🏆 Strategia Raccomandata: Partire con Moondream2

**Perché**: È il miglior rapporto effort/risultato. Con un singolo prompt puoi ottenere una classificazione ragionevole senza dover definire regole manuali o raccogliere dataset. Se non funziona bene, si può sempre aggiungere MediaPipe come fallback leggero.

**Piano B**: Se moondream2 è troppo lento o impreciso, si può provare il modello 0.5B o passare a un approccio YOLO + regole euristiche.

## 🚀 Quick Start

```bash
# Clona e entra nella directory
cd StudyWatchdog

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
- **OpenCV** — webcam
- **PyTorch + Transformers** — AI models
- **Ruff** — linting/formatting
- **pytest** — testing

## 🗺️ Roadmap

### Phase 1: Foundation
- [ ] Webcam capture funzionante (preview live)
- [ ] Struttura progetto e config
- [ ] Entry point CLI

### Phase 2: Detection
- [ ] Integrazione moondream2
- [ ] Classificazione base "studia vs non studia"
- [ ] Benchmark performance su hardware target

### Phase 3: Alerts
- [ ] Riproduzione suono quando ti distrai
- [ ] Timeout e cooldown configurabili
- [ ] TTS base

### Phase 4: Polish
- [ ] System tray / mini GUI
- [ ] Statistiche sessione (% tempo studio)
- [ ] Fine-tune della detection
- [ ] Sistema di escalation degli alert

## 📄 License
MIT
