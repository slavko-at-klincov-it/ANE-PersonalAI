# Personal AI — Lokaler KI-Assistent auf Apple Neural Engine

Ein persönlicher KI-Assistent der **auf deinem Mac lernt** — komplett lokal, ohne Cloud, ohne Kosten. Scannt deine Dateien, lernt deine Patterns, und hilft dir beim Suchen und Arbeiten.

Trainiert ein 109M-Parameter Transformer-Modell direkt auf Apples Neural Engine (ANE) via reverse-engineered private APIs. Läuft auf jedem Apple Silicon Mac (M1-M4).

## Was es tut

```
Deine Dateien  →  Sammeln  →  Tokenisieren  →  ANE Training  →  Suche/Query
(Code, Docs,     (FSEvents)   (BPE/tiktoken)  (91ms/step!)    (Keyword jetzt,
 Notizen...)                                                    Neural später)
```

**Getestet auf:** MacBook Pro M3 Pro, 18GB RAM, macOS 26.3.1

## Quickstart

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install tiktoken watchdog

# 2. ANE Training-Code holen (Dependency)
git clone https://github.com/maderix/ANE.git repo
cd repo/training/training_dynamic && make MODEL=stories110m && cd ../../..

# 3. Dateien sammeln
./pai scan

# 4. Tokenisieren
./pai tokenize

# 5. Trainieren
./pai train

# 6. Suchen
./pai query "was habe ich letzte woche gemacht"
```

## Befehle

```
pai scan [--watch dir]    Dateien scannen und Corpus aktualisieren
pai watch                 Live-Überwachung (sofortige Erkennung neuer Dateien)
pai tokenize              Corpus → Trainings-Tokens (BPE)
pai train                 Training-Session starten (prüft Strom)
pai query [text]          Interaktiv suchen oder Einzel-Query
pai stats                 Status: Corpus, Tokens, Modell, Training
pai recent                Zuletzt gesammelte Dateien zeigen
pai install               Nightly Training aktivieren (2 Uhr, nur wenn eingesteckt)
pai uninstall             Nightly Training deaktivieren
```

## Architektur

### Übersicht

```
┌─────────────────────────────────────────────────────────┐
│                    pai (CLI)                              │
├──────────┬──────────┬──────────────┬────────────────────┤
│ Collector│Tokenizer │   Trainer    │    Inference        │
│          │          │              │                     │
│ FSEvents │ tiktoken │ ANE Training │ Corpus-Suche        │
│ → JSONL  │ BPE      │ 109M params  │ (Neural geplant)   │
│ corpus   │ → uint16 │ 91ms/step    │                     │
├──────────┴──────────┴──────┬───────┴────────────────────┤
│           maderix/ANE      │        libane               │
│     (Training Pipeline)    │   (ANE C-API, optional)     │
├────────────────────────────┴────────────────────────────┤
│        Apple Neural Engine (Private APIs)                │
│    _ANEInMemoryModel, _ANERequest, IOSurface            │
├─────────────────────────────────────────────────────────┤
│              ANE Hardware (M1-M4, 16 Cores)              │
└─────────────────────────────────────────────────────────┘
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

#### 3. Trainer (`trainer/train_nightly.sh`)

Orchestriert den Training-Prozess:

```
1. Ist Laptop eingesteckt? (kein Training auf Batterie)
2. Lock-File prüfen (kein paralleler Run)
3. Collector laufen lassen (neue Dateien sammeln)
4. Tokenizer laufen lassen (neue Tokens erstellen)
5. Training starten:
   - Checkpoint vorhanden? → Resume (inkrementell!)
   - Kein Checkpoint? → From Scratch
6. Checkpoint speichern
```

**Nightly Schedule:** launchd-Agent, läuft um 2:00 Uhr mit Background-Priorität.

**ANE Training Details:**
- Modell: Stories110M (109.5M Parameter, 12 Layers, 768 dim)
- Geschwindigkeit: 91ms/step (M3 Pro, mit Vocab-Compaction)
- 10 ANE-Kernel werden einmal kompiliert (520ms), dann wiederverwendet
- Weights werden im IOSurface-Spatial-Dimension gepackt (kein Recompile bei Update!)
- QoS=9 (Background) — schnellster QoS-Level, niedrigste Systembelastung

#### 4. Inference (`inference/query.py`)

**Jetzt:** Keyword-Suche im Corpus mit Snippet-Anzeige.

**Geplant:** Neural Text-Generation und semantische Suche nach genug Training.

Befehle im interaktiven Modus:
```
/stats          System-Status
/search text    Keyword-Suche
/recent         Letzte Dateien
/quit           Beenden
```

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

## Dependencies

| Dependency | Was | Woher |
|-----------|-----|-------|
| [maderix/ANE](https://github.com/maderix/ANE) | ANE Training-Pipeline | `git clone` in `repo/` |
| [tiktoken](https://pypi.org/project/tiktoken/) | BPE-Tokenizer | `pip install tiktoken` |
| [watchdog](https://pypi.org/project/watchdog/) | FSEvents File-Watcher | `pip install watchdog` |
| Xcode CLI Tools | Compiler für ANE-Kernels | `xcode-select --install` |

Optional:
| [libane](https://github.com/TODO) | Unsere ANE C-API | Separates Repo |

## Worauf baut das auf?

Dieses Projekt ist das Ergebnis eines Forschungsprojekts über Apples Neural Engine:

1. **Reverse Engineering:** 35 private API-Klassen entdeckt, Hardware-Identität probed (M3 Pro = `h15g`, 16 Cores), 6 QoS-Level gefunden (Background=9 ist 42% schneller)
2. **libane:** Eigene C-API mit Version-Detection gebaut (überlebt Apple API-Änderungen)
3. **Benchmarks:** M3 Pro ANE auf 9.36 TFLOPS (FP16) / 18.23 TOPS gemessen, INT8 lohnt nicht
4. **Training verifiziert:** Stories110M auf M3 Pro, 50 Steps stabil, kein NaN

Die vollständige Forschungsdokumentation liegt im [ANE-Training](https://github.com/TODO) Repo.

## Limitierungen

- **Kein neuronales Inference** — aktuell nur Keyword-Suche (Neural kommt nach Training)
- **Kleine Modelle** — 109M Parameter lernt Patterns, aber kein allgemeines Reasoning
- **Private APIs** — können sich mit macOS-Updates ändern (libane hat Version-Detection)
- **Tokenizer-Mismatch** — tiktoken hat 100K Vocab, Modell erwartet 32K (Vocab-Compaction kompensiert)

## Lizenz

MIT
