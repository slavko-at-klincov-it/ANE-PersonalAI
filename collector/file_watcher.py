#!/usr/bin/env python3
"""File watcher daemon — collects new/changed text from your filesystem.
Uses macOS FSEvents for efficient native file monitoring.

Watches configured directories for text file changes, extracts content,
and appends to a training corpus file for the ANE trainer.

Supports plain text, code, PDF, RTF, DOCX, and Apple Mail (.emlx).

Usage:
    python3 file_watcher.py                    # Run with defaults
    python3 file_watcher.py --watch ~/Documents --watch ~/Code
    python3 file_watcher.py --daemon           # Run as background daemon
"""

import os
import sys
import time
import json
import email
import hashlib
import argparse
import signal
import subprocess
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

# Plain text and code files (read directly)
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

# Rich formats requiring extraction (extension → extractor function name)
RICH_EXTENSIONS = {
    '.rtf': 'textutil',
    '.doc': 'textutil',
    '.docx': 'textutil',
    '.odt': 'textutil',
    '.pdf': 'pdf',
    '.emlx': 'emlx',
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
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
LOG_FILE = os.path.join(DATA_DIR, "watcher.log")
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB max per file
MAX_PDF_SIZE = 10 * 1024 * 1024  # 10MB max for PDFs
MAX_CORPUS_SIZE = 500 * 1024 * 1024  # 500MB corpus limit


# ===== Rich Text Extraction =====

def extract_via_textutil(filepath):
    """Extract text from RTF/DOC/DOCX/ODT using macOS built-in textutil."""
    try:
        result = subprocess.run(
            ['textutil', '-convert', 'txt', '-stdout', filepath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def extract_pdf(filepath):
    """Extract text from PDF. Tries PyPDF2, then pdftotext."""
    try:
        import PyPDF2
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if pages:
                return '\n\n'.join(pages)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: pdftotext (from poppler, brew install poppler)
    try:
        result = subprocess.run(
            ['pdftotext', filepath, '-'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def extract_emlx(filepath):
    """Extract text from Apple Mail .emlx files."""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()

        # .emlx starts with a line containing the byte count of the message
        content_str = content.decode('utf-8', errors='replace')
        lines = content_str.split('\n', 1)
        if len(lines) < 2:
            return None

        raw_email = lines[1]

        # Strip trailing Apple plist metadata
        plist_marker = '<?xml version='
        plist_idx = raw_email.find(plist_marker)
        if plist_idx > 0:
            raw_email = raw_email[:plist_idx]

        # Parse email
        msg = email.message_from_string(raw_email)

        # Extract subject and text body
        parts = []
        subject = msg.get('Subject', '')
        if subject:
            parts.append(f"Subject: {subject}")

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode('utf-8', errors='replace'))
        else:
            if msg.get_content_type() == 'text/plain':
                payload = msg.get_payload(decode=True)
                if payload:
                    parts.append(payload.decode('utf-8', errors='replace'))

        return '\n'.join(parts) if parts else None
    except Exception:
        return None


EXTRACTORS = {
    'textutil': extract_via_textutil,
    'pdf': extract_pdf,
    'emlx': extract_emlx,
}


# ===== Configuration Loading =====

def load_config():
    """Load configuration from config.json if it exists."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def get_watch_dirs_from_config():
    """Get enabled watch directories from config.json."""
    config = load_config()
    if config:
        dirs = []
        for source in config.get('sources', []):
            if source.get('enabled', False):
                path = os.path.expanduser(source.get('path', ''))
                if os.path.isdir(path):
                    dirs.append(path)
        if dirs:
            return dirs
    return [d for d in DEFAULT_WATCH_DIRS if os.path.isdir(d)]


def get_enabled_extensions():
    """Get enabled file extensions based on config.json file type settings."""
    config = load_config()
    extensions = set()
    rich = {}

    if config:
        ft = config.get('fileTypes', {})
        if ft.get('plainText', True):
            extensions.update({'.txt', '.md', '.markdown', '.rst', '.org', '.tex',
                             '.json', '.yaml', '.yml', '.toml', '.xml',
                             '.conf', '.cfg', '.ini', '.env.example',
                             '.csv', '.sql'})
        if ft.get('code', True):
            extensions.update({'.py', '.js', '.ts', '.swift', '.m', '.h', '.c', '.cpp', '.rs',
                             '.go', '.java', '.rb', '.sh', '.zsh', '.bash', '.fish',
                             '.html', '.css'})
        if ft.get('richText', True):
            for ext in ['.rtf', '.doc']:
                rich[ext] = 'textutil'
        if ft.get('pdf', True):
            rich['.pdf'] = 'pdf'
        if ft.get('office', True):
            for ext in ['.docx', '.odt']:
                rich[ext] = 'textutil'
        if ft.get('email', False):
            rich['.emlx'] = 'emlx'
    else:
        extensions = TEXT_EXTENSIONS.copy()
        rich = {k: v for k, v in RICH_EXTENSIONS.items() if k != '.emlx'}

    return extensions, rich


# ===== State Tracking =====

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
    """Read text content from a plain text file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def extract_rich_text(filepath, extractor_name):
    """Extract text from a rich format file."""
    extractor = EXTRACTORS.get(extractor_name)
    if extractor:
        return extractor(filepath)
    return None


def collect_file(filepath, state, corpus_fh, text_exts, rich_exts):
    """Process a single file: extract text and append to corpus."""
    if not state.needs_update(filepath):
        return False

    ext = os.path.splitext(filepath)[1].lower()

    # Determine max size and extraction method
    if ext in rich_exts:
        max_size = MAX_PDF_SIZE if ext == '.pdf' else MAX_FILE_SIZE
        if os.path.getsize(filepath) > max_size:
            return False
        if is_sensitive(filepath):
            return False
        text = extract_rich_text(filepath, rich_exts[ext])
    elif ext in text_exts:
        if os.path.getsize(filepath) > MAX_FILE_SIZE:
            return False
        if is_sensitive(filepath):
            return False
        text = extract_text(filepath)
    else:
        return False

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


def scan_directory(dirpath, state, corpus_fh, stats, text_exts, rich_exts):
    """Recursively scan a directory for text files."""
    all_exts = text_exts | set(rich_exts.keys())
    try:
        entries = os.scandir(dirpath)
    except PermissionError:
        return

    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            if not should_skip_dir(entry.name):
                scan_directory(entry.path, state, corpus_fh, stats, text_exts, rich_exts)
        elif entry.is_file(follow_symlinks=False):
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in all_exts:
                stats['scanned'] += 1
                if collect_file(entry.path, state, corpus_fh, text_exts, rich_exts):
                    stats['collected'] += 1


def full_scan(watch_dirs, state):
    """Do a full scan of all watched directories."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check corpus size
    if os.path.exists(CORPUS_FILE) and os.path.getsize(CORPUS_FILE) > MAX_CORPUS_SIZE:
        print(f"Corpus at {MAX_CORPUS_SIZE/1e6:.0f}MB limit. Rotate or train first.")
        return

    text_exts, rich_exts = get_enabled_extensions()

    stats = {'scanned': 0, 'collected': 0}
    with open(CORPUS_FILE, 'a', encoding='utf-8') as fh:
        for d in watch_dirs:
            d = os.path.expanduser(d)
            if os.path.isdir(d):
                print(f"Scanning {d}...")
                scan_directory(d, state, fh, stats, text_exts, rich_exts)

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
            self.text_exts, self.rich_exts = get_enabled_extensions()
            self.all_exts = self.text_exts | set(self.rich_exts.keys())

        def on_modified(self, event):
            self._handle(event.src_path)

        def on_created(self, event):
            self._handle(event.src_path)

        def _handle(self, filepath):
            if not os.path.isfile(filepath):
                return
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in self.all_exts:
                return
            if any(skip in filepath for skip in SKIP_PATTERNS):
                return
            if collect_file(filepath, self.state, self.corpus_fh,
                          self.text_exts, self.rich_exts):
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

    # Use config.json sources if no --watch specified
    if args.watch:
        watch_dirs = args.watch
    else:
        watch_dirs = get_watch_dirs_from_config()

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
