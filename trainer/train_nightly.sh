#!/bin/bash
# train_nightly.sh — Nightly ANE training scheduler
# Runs a larger batch of training when the laptop is plugged in and idle.
# Complement to "pai learn" (continuous daytime training).
#
# Install as launchd agent:
#   cp com.personal-ai.train.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.personal-ai.train.plist

set -e

DATA_DIR="$HOME/.local/personal-ai"
# ANE Training platform
ANE_DIR="$HOME/Code/ANE-Training"
TRAIN_DIR="$ANE_DIR/training/training_dynamic"
LOG_FILE="$DATA_DIR/train.log"
LOCK_FILE="$DATA_DIR/train.lock"
CORPUS_FILE="$DATA_DIR/corpus.jsonl"
TRAINING_DATA="$DATA_DIR/training_data.bin"

mkdir -p "$DATA_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; echo "$*"; }

# ===== Checks =====

# Check if already running
if [ -f "$LOCK_FILE" ]; then
    pid=$(cat "$LOCK_FILE" 2>/dev/null)
    if kill -0 "$pid" 2>/dev/null; then
        log "Training already running (pid=$pid), skipping"
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi

# Check if plugged in (don't train on battery)
if pmset -g ps | grep -q "Battery Power"; then
    log "On battery power, skipping training"
    exit 0
fi

# Check if corpus has new data
if [ ! -f "$CORPUS_FILE" ]; then
    log "No corpus file, running collector first"
    python3 "$(dirname "$0")/../collector/file_watcher.py" --scan 2>&1 | tee -a "$LOG_FILE"
fi

# ===== Prepare training data =====

log "Preparing training data..."
python3 "$(dirname "$0")/../tokenizer/prepare_training_data.py" \
    --corpus "$CORPUS_FILE" \
    --output "$TRAINING_DATA" \
    2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$TRAINING_DATA" ]; then
    log "Failed to create training data"
    exit 1
fi

DATA_SIZE=$(stat -f%z "$TRAINING_DATA" 2>/dev/null || stat -c%s "$TRAINING_DATA")
log "Training data: $DATA_SIZE bytes"

# Need at least 100KB of data
if [ "$DATA_SIZE" -lt 102400 ]; then
    log "Not enough training data ($DATA_SIZE bytes < 100KB), skipping"
    exit 0
fi

# ===== Run training =====

echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

log "Starting ANE nightly training..."

cd "$TRAIN_DIR"

# Build if needed
if [ ! -f train ] || [ train.m -nt train ]; then
    log "Building trainer..."
    make MODEL=stories110m 2>&1 | tee -a "$LOG_FILE"
fi

# Symlink training data
ln -sf "$TRAINING_DATA" "$ANE_DIR/training/tinystories_data00.bin"

# Run training (500 steps per nightly session — larger batch than daytime)
STEPS=500
CKPT="$DATA_DIR/checkpoint.bin"

if [ -f "$CKPT" ]; then
    # Incremental: resume from last checkpoint
    log "Resuming training from checkpoint (step $(python3 -c "
import struct
with open('$CKPT','rb') as f:
    f.read(8)
    print(struct.unpack('<I',f.read(4))[0])
" 2>/dev/null || echo '?'))"
    cp "$CKPT" checkpoint.bin 2>/dev/null
    ./train --resume --steps "$STEPS" --accum 5 --data "$TRAINING_DATA" \
        2>&1 | tee -a "$LOG_FILE"
else
    # First run: train from scratch
    log "First training run (from scratch)"
    ./train --scratch --steps "$STEPS" --accum 5 --data "$TRAINING_DATA" \
        2>&1 | tee -a "$LOG_FILE"
fi

# Save checkpoint back to data dir
if [ -f checkpoint.bin ]; then
    cp checkpoint.bin "$CKPT"
    log "Checkpoint saved to $CKPT"
fi

log "Nightly training complete"

# ===== Cleanup =====
rm -f "$LOCK_FILE"
