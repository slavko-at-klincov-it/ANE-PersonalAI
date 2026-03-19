#!/usr/bin/env python3
"""Continuous ANE trainer — watches files and trains incrementally all day.

Runs as a background process with zero impact: no fan, no battery drain,
GPU/CPU stay free. Uses ANE QoS=9 (Background) for training.

Architecture:
  1. File watcher detects changes in configured folders (FSEvents)
  2. On change: tokenize new content, train 10-50 steps on ANE
  3. Model is saved after each mini-batch (survives restarts)
  4. Runs all day without manual intervention

Usage:
    python3 continuous_trainer.py              # Run in foreground
    python3 continuous_trainer.py --daemon      # Run as background daemon
    python3 continuous_trainer.py --stop        # Stop running daemon
    python3 continuous_trainer.py --status      # Show daemon status
"""

import os
import sys
import time
import json
import signal
import hashlib
import struct
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

# ===== Configuration =====

DATA_DIR = os.path.expanduser("~/.local/personal-ai")
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
TRAINING_DATA = os.path.join(DATA_DIR, "training_data.bin")
CHECKPOINT = os.path.join(DATA_DIR, "checkpoint.bin")
PID_FILE = os.path.join(DATA_DIR, "learn.pid")
LOG_FILE = os.path.join(DATA_DIR, "learn.log")
STATE_FILE = os.path.join(DATA_DIR, "learn_state.json")

# ANE-Training repo location
ANE_TRAINING_DIR = os.path.expanduser("~/Code/ANE-Training")
TRAIN_BIN_DIR = os.path.join(ANE_TRAINING_DIR, "training", "training_dynamic")

# Training parameters
MIN_STEPS = 10          # Minimum steps per mini-batch
MAX_STEPS = 50          # Maximum steps per mini-batch
ACCUM_STEPS = 5         # Gradient accumulation
DEBOUNCE_SECS = 30      # Wait after last file change before training
MIN_DATA_BYTES = 51200  # 50KB minimum training data

# Watch directories (same as collector)
DEFAULT_WATCH_DIRS = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Code"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Notes"),
]

CONFIG_FILE = os.path.join(DATA_DIR, "config.json")


def load_watch_dirs():
    """Load watch directories from config.json, falling back to defaults."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            dirs = []
            for source in config.get('sources', []):
                if source.get('enabled', False):
                    path = os.path.expanduser(source.get('path', ''))
                    if os.path.isdir(path):
                        dirs.append(path)
            if dirs:
                return dirs
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return [d for d in DEFAULT_WATCH_DIRS if os.path.isdir(d)]

# File extensions to watch
TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown', '.rst', '.org', '.tex',
    '.py', '.js', '.ts', '.swift', '.m', '.h', '.c', '.cpp', '.rs',
    '.go', '.java', '.rb', '.sh', '.zsh', '.bash',
    '.html', '.css', '.json', '.yaml', '.yml', '.toml', '.xml',
    '.conf', '.cfg', '.ini', '.csv', '.sql',
}

SKIP_PATTERNS = {
    '.git', '.svn', 'node_modules', '__pycache__', '.venv', 'venv',
    '.DS_Store', '.Trash', 'Library', '.cache', '.npm', '.cargo',
    'build', 'dist', '.next', '.nuxt', 'target',
}


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except OSError:
        pass


class LearnState:
    """Persistent state for continuous learning."""

    def __init__(self):
        self.total_steps = 0
        self.total_batches = 0
        self.last_train_time = None
        self.started_at = None
        self.load()

    def load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                self.total_steps = data.get('total_steps', 0)
                self.total_batches = data.get('total_batches', 0)
                self.last_train_time = data.get('last_train_time')
                self.started_at = data.get('started_at')
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        data = {
            'total_steps': self.total_steps,
            'total_batches': self.total_batches,
            'last_train_time': self.last_train_time,
            'started_at': self.started_at,
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def record_batch(self, steps):
        self.total_steps += steps
        self.total_batches += 1
        self.last_train_time = datetime.now().isoformat()
        self.save()


class ChangeAccumulator:
    """Accumulates file changes and debounces before triggering training."""

    def __init__(self):
        self.changed_files = set()
        self.last_change_time = 0

    def add(self, filepath):
        self.changed_files.add(filepath)
        self.last_change_time = time.time()

    def ready(self):
        """True if we have changes and debounce period has passed."""
        if not self.changed_files:
            return False
        return (time.time() - self.last_change_time) >= DEBOUNCE_SECS

    def consume(self):
        """Return and clear accumulated changes."""
        files = self.changed_files.copy()
        self.changed_files.clear()
        return files


if HAS_WATCHDOG:
    class LearnFileHandler(FileSystemEventHandler):
        """Watches for file changes and feeds them to the accumulator."""

        def __init__(self, accumulator):
            self.accumulator = accumulator

        def on_modified(self, event):
            self._handle(event.src_path)

        def on_created(self, event):
            self._handle(event.src_path)

        def _handle(self, filepath):
            if not os.path.isfile(filepath):
                return
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in TEXT_EXTENSIONS:
                return
            if any(skip in filepath for skip in SKIP_PATTERNS):
                return
            self.accumulator.add(filepath)


def collect_and_tokenize():
    """Run collector and tokenizer to update training data."""
    pai_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Collect new files
    collector = os.path.join(pai_dir, 'collector', 'file_watcher.py')
    subprocess.run(
        [sys.executable, collector, '--scan'],
        capture_output=True, timeout=120
    )

    # Tokenize
    tokenizer = os.path.join(pai_dir, 'tokenizer', 'prepare_training_data.py')
    subprocess.run(
        [sys.executable, tokenizer],
        capture_output=True, timeout=120
    )


def run_training(steps):
    """Run a mini-batch of ANE training steps."""
    train_bin = os.path.join(TRAIN_BIN_DIR, 'train')

    if not os.path.exists(train_bin):
        log(f"Training binary not found: {train_bin}")
        log("Build with: cd ANE-Training/training/training_dynamic && make MODEL=stories110m")
        return False

    if not os.path.exists(TRAINING_DATA):
        log("No training data available")
        return False

    data_size = os.path.getsize(TRAINING_DATA)
    if data_size < MIN_DATA_BYTES:
        log(f"Not enough training data ({data_size} bytes < {MIN_DATA_BYTES})")
        return False

    # Symlink training data where trainer expects it
    data_link = os.path.join(os.path.dirname(TRAIN_BIN_DIR), 'tinystories_data00.bin')
    try:
        if os.path.islink(data_link) or os.path.exists(data_link):
            os.remove(data_link)
        os.symlink(TRAINING_DATA, data_link)
    except OSError as e:
        log(f"Symlink failed: {e}")
        return False

    # Copy checkpoint to training dir if exists
    train_ckpt = os.path.join(TRAIN_BIN_DIR, 'checkpoint.bin')
    if os.path.exists(CHECKPOINT):
        subprocess.run(['cp', CHECKPOINT, train_ckpt])
        args = [train_bin, '--resume', '--steps', str(steps), '--accum', str(ACCUM_STEPS),
                '--data', TRAINING_DATA]
    else:
        args = [train_bin, '--scratch', '--steps', str(steps), '--accum', str(ACCUM_STEPS),
                '--data', TRAINING_DATA]

    log(f"Training {steps} steps...")
    try:
        result = subprocess.run(
            args, cwd=TRAIN_BIN_DIR,
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            # Save checkpoint back
            if os.path.exists(train_ckpt):
                subprocess.run(['cp', train_ckpt, CHECKPOINT])
            log(f"Mini-batch complete ({steps} steps)")
            return True
        else:
            log(f"Training failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        log("Training timed out")
        return False


def calculate_steps(n_changed_files):
    """Determine steps based on amount of new content."""
    # More changed files → more steps, capped at MAX_STEPS
    steps = min(MIN_STEPS + n_changed_files * 2, MAX_STEPS)
    return steps


def run_learn_loop(watch_dirs):
    """Main continuous learning loop."""
    if not HAS_WATCHDOG:
        log("Error: watchdog not installed. Run: pip install watchdog")
        return

    state = LearnState()
    state.started_at = datetime.now().isoformat()
    state.save()

    accumulator = ChangeAccumulator()
    handler = LearnFileHandler(accumulator)
    observer = Observer()

    for d in watch_dirs:
        if os.path.isdir(d):
            observer.schedule(handler, d, recursive=True)
            log(f"Watching: {d}")

    observer.start()

    # Initial collection
    log("Initial scan...")
    collect_and_tokenize()

    log("Continuous learning active. Zero impact — ANE QoS=9 (Background)")
    log("Model saved after each mini-batch. Ctrl+C to stop.")

    try:
        while True:
            if accumulator.ready():
                changed = accumulator.consume()
                n_files = len(changed)
                log(f"{n_files} file(s) changed, collecting & training...")

                collect_and_tokenize()
                steps = calculate_steps(n_files)

                if run_training(steps):
                    state.record_batch(steps)
                    log(f"Session: {state.total_batches} batches, "
                        f"{state.total_steps} total steps today")

            time.sleep(5)
    except KeyboardInterrupt:
        log("Stopping continuous learning...")
        observer.stop()
    observer.join()

    log(f"Session summary: {state.total_batches} batches, {state.total_steps} steps")


def daemonize():
    """Fork into background daemon."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)

    # Redirect stdio
    sys.stdin = open(os.devnull, 'r')
    sys.stdout = open(LOG_FILE, 'a')
    sys.stderr = sys.stdout

    # Write PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))


def stop_daemon():
    """Stop running daemon."""
    if not os.path.exists(PID_FILE):
        print("No learn daemon running")
        return
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        os.remove(PID_FILE)
        print(f"Stopped learn daemon (pid={pid})")
    except (ProcessLookupError, ValueError):
        os.remove(PID_FILE)
        print("Daemon was not running (stale PID file removed)")


def show_status():
    """Show daemon status."""
    running = False
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            running = True
            print(f"Learn daemon: RUNNING (pid={pid})")
        except (ProcessLookupError, ValueError):
            print("Learn daemon: NOT RUNNING (stale PID)")

    if not running and not os.path.exists(PID_FILE):
        print("Learn daemon: NOT RUNNING")

    state = LearnState()
    if state.total_batches > 0:
        print(f"Total batches: {state.total_batches}")
        print(f"Total steps:   {state.total_steps}")
        if state.last_train_time:
            print(f"Last trained:  {state.last_train_time}")
        if state.started_at:
            print(f"Started at:    {state.started_at}")
    else:
        print("No training sessions yet")


def main():
    parser = argparse.ArgumentParser(description='Continuous ANE trainer')
    parser.add_argument('--daemon', action='store_true', help='Run as background daemon')
    parser.add_argument('--stop', action='store_true', help='Stop running daemon')
    parser.add_argument('--status', action='store_true', help='Show daemon status')
    parser.add_argument('--watch', action='append', help='Directory to watch')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.stop:
        stop_daemon()
        return

    if args.status:
        show_status()
        return

    watch_dirs = args.watch or load_watch_dirs()

    if args.daemon:
        print("Starting continuous learning daemon...")
        print(f"  Log: {LOG_FILE}")
        print(f"  PID: {PID_FILE}")
        print("  Stop with: pai learn --stop")
        daemonize()

    run_learn_loop(watch_dirs)


if __name__ == '__main__':
    main()
