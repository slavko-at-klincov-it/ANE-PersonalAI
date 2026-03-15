#!/usr/bin/env python3
"""File watcher daemon — collects new/changed text from your filesystem.
Uses macOS FSEvents for efficient native file monitoring.

Watches configured directories for text file changes, extracts content,
and appends to a training corpus file for the ANE trainer.

Usage:
    python3 file_watcher.py                    # Run with defaults
    python3 file_watcher.py --watch ~/Documents --watch ~/Code
    python3 file_watcher.py --daemon           # Run as background daemon
"""

import os
import sys
import time
import json
import hashlib
import argparse
import signal
from pathlib import Path
from datetime import datetime

# FSEvents via fsevents (pip install fsevents) or fallback to watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

# ===== Configuration =====

DEFAULT_WATCH_DIRS = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Code"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Notes"),
]

# File extensions to collect text from
TEXT_EXTENSIONS = {
    # Documents
    '.txt', '.md', '.markdown', '.rst', '.org', '.tex',
    # Code
    '.py', '.js', '.ts', '.swift', '.m', '.h', '.c', '.cpp', '.rs',
    '.go', '.java', '.rb', '.sh', '.zsh', '.bash', '.fish',
    '.html', '.css', '.json', '.yaml', '.yml', '.toml', '.xml',
    # Config
    '.conf', '.cfg', '.ini', '.env.example',
    # Data
    '.csv', '.sql',
}

# Files/dirs to skip
SKIP_PATTERNS = {
    '.git', '.svn', 'node_modules', '__pycache__', '.venv', 'venv',
    '.DS_Store', '.Trash', 'Library', '.cache', '.npm', '.cargo',
    'build', 'dist', '.next', '.nuxt', 'target',
}

# Sensitive patterns to NEVER collect
SENSITIVE_PATTERNS = {
    '.env', '.key', '.pem', '.p12', '.pfx', '.keystore',
    'credentials', 'secret', 'password', 'token', 'api_key',
    'id_rsa', 'id_ed25519', '.ssh',
}

DATA_DIR = os.path.expanduser("~/.local/personal-ai")
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
STATE_FILE = os.path.join(DATA_DIR, "watcher_state.json")
LOG_FILE = os.path.join(DATA_DIR, "watcher.log")
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB max per file
MAX_CORPUS_SIZE = 500 * 1024 * 1024  # 500MB corpus limit


class WatcherState:
    """Tracks which files have been processed and their hashes."""

    def __init__(self, path):
        self.path = path
        self.files = {}  # path -> {hash, mtime, size, collected_at}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.files = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.files, f, indent=2)

    def needs_update(self, filepath):
        """Check if file is new or changed since last collection."""
        try:
            stat = os.stat(filepath)
        except OSError:
            return False

        key = str(filepath)
        if key not in self.files:
            return True

        entry = self.files[key]
        return stat.st_mtime > entry.get('mtime', 0)

    def mark_collected(self, filepath, content_hash):
        key = str(filepath)
        stat = os.stat(filepath)
        self.files[key] = {
            'hash': content_hash,
            'mtime': stat.st_mtime,
            'size': stat.st_size,
            'collected_at': datetime.now().isoformat(),
        }


def is_sensitive(filepath):
    """Check if file might contain secrets."""
    name = os.path.basename(filepath).lower()
    path_lower = filepath.lower()
    for pat in SENSITIVE_PATTERNS:
        if pat in name or pat in path_lower:
            return True
    return False


def should_skip_dir(dirname):
    """Check if directory should be skipped."""
    return dirname in SKIP_PATTERNS or dirname.startswith('.')


def extract_text(filepath):
    """Read text content from a file, handling encoding gracefully."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def collect_file(filepath, state, corpus_fh):
    """Process a single file: extract text and append to corpus."""
    if not state.needs_update(filepath):
        return False

    if os.path.getsize(filepath) > MAX_FILE_SIZE:
        return False

    if is_sensitive(filepath):
        return False

    text = extract_text(filepath)
    if not text or len(text.strip()) < 10:
        return False

    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    # Check if content actually changed (not just mtime)
    key = str(filepath)
    if key in state.files and state.files[key].get('hash') == content_hash:
        state.files[key]['mtime'] = os.stat(filepath).st_mtime
        return False

    # Write to corpus as JSONL
    entry = {
        'path': str(filepath),
        'text': text,
        'hash': content_hash,
        'collected_at': datetime.now().isoformat(),
        'size': len(text),
    }
    corpus_fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    corpus_fh.flush()

    state.mark_collected(filepath, content_hash)
    return True


def scan_directory(dirpath, state, corpus_fh, stats):
    """Recursively scan a directory for text files."""
    try:
        entries = os.scandir(dirpath)
    except PermissionError:
        return

    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            if not should_skip_dir(entry.name):
                scan_directory(entry.path, state, corpus_fh, stats)
        elif entry.is_file(follow_symlinks=False):
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in TEXT_EXTENSIONS:
                stats['scanned'] += 1
                if collect_file(entry.path, state, corpus_fh):
                    stats['collected'] += 1


def full_scan(watch_dirs, state):
    """Do a full scan of all watched directories."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check corpus size
    if os.path.exists(CORPUS_FILE) and os.path.getsize(CORPUS_FILE) > MAX_CORPUS_SIZE:
        print(f"Corpus at {MAX_CORPUS_SIZE/1e6:.0f}MB limit. Rotate or train first.")
        return

    stats = {'scanned': 0, 'collected': 0}
    with open(CORPUS_FILE, 'a', encoding='utf-8') as fh:
        for d in watch_dirs:
            d = os.path.expanduser(d)
            if os.path.isdir(d):
                print(f"Scanning {d}...")
                scan_directory(d, state, fh, stats)

    state.save()
    corpus_size = os.path.getsize(CORPUS_FILE) if os.path.exists(CORPUS_FILE) else 0
    print(f"Scanned {stats['scanned']} files, collected {stats['collected']} new/changed")
    print(f"Corpus: {corpus_size/1e6:.1f} MB ({CORPUS_FILE})")


# ===== Live watcher using watchdog =====

if HAS_WATCHDOG:
    class TextFileHandler(FileSystemEventHandler):
        def __init__(self, state):
            self.state = state
            self.corpus_fh = open(CORPUS_FILE, 'a', encoding='utf-8')

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
            if collect_file(filepath, self.state, self.corpus_fh):
                name = os.path.basename(filepath)
                print(f"  + {name}")
                self.state.save()


def run_live(watch_dirs, state):
    """Run live file watcher."""
    if not HAS_WATCHDOG:
        print("Install watchdog for live monitoring: pip3 install watchdog")
        print("Falling back to periodic scan (every 5 minutes)")
        while True:
            full_scan(watch_dirs, state)
            time.sleep(300)
        return

    handler = TextFileHandler(state)
    observer = Observer()
    for d in watch_dirs:
        d = os.path.expanduser(d)
        if os.path.isdir(d):
            observer.schedule(handler, d, recursive=True)
            print(f"Watching: {d}")

    observer.start()
    print("Live file watcher running. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    handler.state.save()


def main():
    parser = argparse.ArgumentParser(description='Personal AI file collector')
    parser.add_argument('--watch', action='append', help='Directory to watch')
    parser.add_argument('--scan', action='store_true', help='Do a one-time scan')
    parser.add_argument('--live', action='store_true', help='Watch for changes in real-time')
    parser.add_argument('--stats', action='store_true', help='Show corpus statistics')
    args = parser.parse_args()

    watch_dirs = args.watch or [d for d in DEFAULT_WATCH_DIRS if os.path.isdir(d)]
    state = WatcherState(STATE_FILE)

    if args.stats:
        if os.path.exists(CORPUS_FILE):
            size = os.path.getsize(CORPUS_FILE)
            lines = sum(1 for _ in open(CORPUS_FILE))
            print(f"Corpus: {size/1e6:.1f} MB, {lines} documents")
            print(f"Tracked files: {len(state.files)}")
        else:
            print("No corpus yet. Run --scan first.")
    elif args.live:
        full_scan(watch_dirs, state)  # Initial scan
        run_live(watch_dirs, state)
    else:
        full_scan(watch_dirs, state)


if __name__ == '__main__':
    main()
