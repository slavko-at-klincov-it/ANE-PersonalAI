# Personal AI — Lokaler KI-Assistent auf Apple Neural Engine

Ein persönlicher KI-Assistent der **auf deinem Mac lernt** — komplett lokal, ohne Cloud, ohne Kosten. Scannt deine Dateien, lernt deine Patterns, und hilft dir beim Suchen und Arbeiten.

Trainiert ein 109M-Parameter Transformer-Modell direkt auf Apples Neural Engine (ANE) via reverse-engineered private APIs. Läuft auf jedem Apple Silicon Mac (M1-M4).

## Was es tut

```
Deine Dateien  →  Sammeln  →  Tokenisieren  →  ANE Training  →  Suche/Query
(Code, Docs,     (FSEvents)   (BPE/tiktoken)  (80.9ms/step)   (Keyword jetzt,
 Notizen...)                                                    Neural später)
```

**Getestet auf:** MacBook Pro M3 Pro, 18GB RAM, macOS 26.3.1

## Kern-Feature: Continuous Learning

```bash
pai learn
```

Einmal starten, den ganzen Tag lernen. **Zero Impact** auf dein System:

- **File Watcher** erkennt Änderungen in konfigurierten Ordnern (FSEvents)
- **Bei Änderung:** tokenisiert neuen Content, trainiert 10-50 Steps auf ANE (QoS=9)
- **Hintergrundprozess** — kein Lüfter, kein Akku-Drain, GPU/CPU bleiben frei
- **Modell wird nach jedem Mini-Batch gespeichert** (überlebt Neustart)
- **Kein manuelles Starten nötig** — einmal `pai learn` und es läuft den ganzen Tag

```
Du arbeitest normal  →  File Watcher erkennt Änderung  →  30s Debounce
                        →  Collect & Tokenize  →  10-50 ANE Steps  →  Checkpoint saved
                        →  Weiter warten...
```

Nightly-Training bleibt als Option für größere Batches über Nacht (500 Steps um 2 Uhr).

## Quickstart

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install tiktoken watchdog

# 2. ANE-Training Platform holen (Dependency)
git clone https://github.com/slavko-at-klincov-it/ANE-Training.git ~/Code/ANE-Training
cd ~/Code/ANE-Training/training/training_dynamic && make MODEL=stories110m && cd -

# 3. Dateien sammeln
./pai scan

# 4. Tokenisieren
./pai tokenize

# 5. Continuous Learning starten
./pai learn

# 6. Suchen
./pai query "was habe ich letzte woche gemacht"
```

## Befehle

```
pai scan [--watch dir]    Dateien scannen und Corpus aktualisieren
pai watch                 Live-Überwachung (sofortige Erkennung neuer Dateien)
pai tokenize              Corpus → Trainings-Tokens (BPE)
pai learn                 Continuous Learning starten (den ganzen Tag, zero impact)
pai learn --daemon        Als Hintergrund-Daemon starten
pai learn --stop          Continuous Learning stoppen
pai learn --status        Status des Learning-Daemons anzeigen
pai train                 Einzelne Training-Session starten
pai query [text]          Interaktiv suchen oder Einzel-Query
pai stats                 Status: Corpus, Tokens, Modell, Training
pai recent                Zuletzt gesammelte Dateien zeigen
pai install               Nightly Training aktivieren (2 Uhr, nur wenn eingesteckt)
pai uninstall             Nightly Training deaktivieren
pai app-install           Menu Bar App bei Login starten
pai app-uninstall         App Auto-Start entfernen
```

## Menu Bar App

Eine native macOS Menu Bar App für komfortables Management:

- **Onboarding** — geführte Ersteinrichtung mit Quellen-Auswahl
- **Quellen verwalten** — Verzeichnisse aktivieren/deaktivieren, eigene hinzufügen
- **Einstellungen** — Dateitypen, Training-Parameter, Nightly Training
- **Service-Kontrolle** — Continuous Learning starten/stoppen direkt aus der Menüleiste

### Bauen & Installieren

```bash
# 1. Bauen
cd app && ./build.sh

# 2. Installieren (optional)
cp -r PersonalAI.app /Applications/

# 3. Starten
open PersonalAI.app
```

Erfordert Xcode CLI Tools (`xcode-select --install`).

## Architektur

### Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                       pai (CLI)                             │
├──────────┬──────────┬───────────────┬───────────────────────┤
│ Collector│Tokenizer │   Trainer     │    Inference          │
│          │          │               │                       │
│ FSEvents │ tiktoken │ Continuous    │ Corpus-Suche          │
│ → JSONL  │ BPE      │ + Nightly    │ (Neural geplant)      │
│ corpus   │ → uint16 │ 80.9ms/step  │                       │
├──────────┴──────────┴──────┬────────┴───────────────────────┤
│    slavko-at-klincov-it/   │                                │
│       ANE-Training         │        libane                  │
│    (Training Pipeline)     │   (ANE C-API)                  │
├────────────────────────────┴────────────────────────────────┤
│           Apple Neural Engine (Private APIs)                │
│       _ANEInMemoryModel, _ANERequest, IOSurface             │
├─────────────────────────────────────────────────────────────┤
│             ANE Hardware (M1-M4, 16 Cores)                  │
└─────────────────────────────────────────────────────────────┘
```

### Komponenten

#### 1. Collector (`collector/file_watcher.py`)

Scannt konfigurierte Verzeichnisse nach Textdateien und sammelt sie in einem JSONL-Corpus.

**Was es scannt:**
- Code: `.py`, `.js`, `.ts`, `.swift`, `.m`, `.h`, `.c`, `.cpp`, `.rs`, `.go`, etc.
- Dokumente: `.txt`, `.md`, `.rst`, `.org`, `.tex`
- Config: `.json`, `.yaml`, `.toml`, `.xml`
- Daten: `.csv`, `.sql`

**Was es NICHT scannt (Sicherheit):**
- `.env`, SSH-Keys, Credentials, Passwörter, API-Keys
- `.git`, `node_modules`, `__pycache__`, Build-Verzeichnisse
- Dateien > 1MB
- Alles in `.Trash`, `Library`, `.cache`

**Wie es funktioniert:**
- Trackt jede Datei via SHA256-Hash + mtime
- Erkennt nur geänderte Dateien (kein Doppel-Sammeln)
- Live-Modus nutzt macOS FSEvents via `watchdog`
- State in `~/.local/personal-ai/watcher_state.json`

#### 2. Tokenizer (`tokenizer/prepare_training_data.py`)

Konvertiert den JSONL-Corpus in binäre Token-IDs (uint16) für den ANE-Trainer.

| Tokenizer | Vocab | Effizienz | Wann |
|-----------|-------|-----------|------|
| tiktoken (cl100k_base) | 100K | 3.2x besser als char | Wenn `pip install tiktoken` |
| Character-Level | ~125 | Baseline | Fallback |

**Vocab-Compaction:** Das Training erkennt automatisch welche Token-IDs aktiv sind und komprimiert den Classifier. Bei 124 aktiven Tokens: Classifier sinkt von 87ms auf 0.4ms pro Step.

#### 3. Continuous Trainer (`trainer/continuous_trainer.py`)

Das Herzstück — lernt den ganzen Tag im Hintergrund:

```
File Watcher (FSEvents)
    │
    ▼
Debounce (30s nach letzter Änderung)
    │
    ▼
Collect → Tokenize → Train (10-50 Steps, ANE QoS=9)
    │
    ▼
Checkpoint saved → weiter warten
```

- **Zero Impact:** ANE QoS=9 (Background), kein Lüfter, kein Akku
- **Adaptiv:** Mehr geänderte Dateien → mehr Training-Steps (10-50)
- **Persistent:** Modell überlebt Neustarts, wird nach jedem Mini-Batch gespeichert
- **Daemon-Modus:** `pai learn --daemon` für unsichtbaren Hintergrundbetrieb

#### 4. Nightly Trainer (`trainer/train_nightly.sh`)

Ergänzt das Continuous Learning mit größeren Batches über Nacht:

- 500 Steps pro Nacht (vs. 10-50 pro Mini-Batch tagsüber)
- Nur wenn eingesteckt (kein Training auf Batterie)
- launchd-Agent, läuft um 2:00 Uhr mit Background-Priorität

#### 5. Inference (`inference/query.py`)

**Jetzt:** Keyword-Suche im Corpus mit Snippet-Anzeige.

**Geplant:** Neural Text-Generation und semantische Suche nach genug Training.

Befehle im interaktiven Modus:
```
/stats          System-Status
/search text    Keyword-Suche
/recent         Letzte Dateien
/quit           Beenden
```

## ANE Performance

**Gemessen auf MacBook Pro M3 Pro (16 ANE Cores):**

| Metrik | Wert |
|--------|------|
| Peak-Performance | **12.79 TFLOPS** (FP16) |
| Training-Durchsatz | **2.15 TFLOPS** |
| Training-Speed | **80.9 ms/step** |
| ANE-Kernels | 10 (einmal kompiliert in 520ms, dann wiederverwendet) |
| Weight-Packing | IOSurface-Spatial-Dimension (kein Recompile bei Update) |
| QoS=9 (Background) | Schnellster QoS-Level, niedrigste Systembelastung |

## Daten

Alles in `~/.local/personal-ai/` (wird NICHT committed):

| Datei | Inhalt |
|-------|--------|
| `corpus.jsonl` | Gesammelte Dokumente (JSONL, ~1MB pro 64 Dateien) |
| `training_data.bin` | Tokenisierte Trainingsdaten (uint16) |
| `watcher_state.json` | Tracking welche Dateien schon gesammelt |
| `tokenizer.json` | Tokenizer-State (nur bei char-level) |
| `checkpoint.bin` | Trainiertes Modell (nach Training) |
| `train.log` | Training-Protokoll |
| `learn.log` | Continuous Learning Log |
| `learn.pid` | PID des Learning-Daemons |
| `learn_state.json` | Continuous Learning Statistiken |

## Dependencies

| Dependency | Was | Woher |
|-----------|-----|-------|
| [ANE-Training](https://github.com/slavko-at-klincov-it/ANE-Training) | ANE Training-Pipeline + libane | `git clone` nach `~/Code/ANE-Training` |
| [tiktoken](https://pypi.org/project/tiktoken/) | BPE-Tokenizer | `pip install tiktoken` |
| [watchdog](https://pypi.org/project/watchdog/) | FSEvents File-Watcher | `pip install watchdog` |
| Xcode CLI Tools | Compiler für ANE-Kernels | `xcode-select --install` |

## Worauf baut das auf?

Dieses Projekt ist das Ergebnis eines Forschungsprojekts über Apples Neural Engine:

1. **Reverse Engineering:** 35 private API-Klassen entdeckt, Hardware-Identität probed (M3 Pro = `h15g`, 16 Cores), 6 QoS-Level gefunden (Background=9 ist 42% schneller)
2. **libane:** Eigene C-API mit Version-Detection gebaut (überlebt Apple API-Änderungen)
3. **Benchmarks:** M3 Pro ANE Peak bei 12.79 TFLOPS (FP16), Training bei 2.15 TFLOPS, 80.9ms/step
4. **Training verifiziert:** Stories110M auf M3 Pro, stabil, kein NaN

Die vollständige Forschungsdokumentation liegt im [ANE-Training](https://github.com/slavko-at-klincov-it/ANE-Training) Repo.

## Limitierungen

- **Kein neuronales Inference** — aktuell nur Keyword-Suche (Neural kommt nach Training)
- **Kleine Modelle** — 109M Parameter lernt Patterns, aber kein allgemeines Reasoning
- **Private APIs** — können sich mit macOS-Updates ändern (libane hat Version-Detection)
- **Tokenizer-Mismatch** — tiktoken hat 100K Vocab, Modell erwartet 32K (Vocab-Compaction kompensiert)

## Lizenz

MIT
