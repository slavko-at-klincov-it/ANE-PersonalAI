"""End-to-end tests for the Personal AI pipeline.

Tests the full flow: scan → tokenize → verify output.
"""

import os
import sys
import json
import struct
import subprocess
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===== Full Pipeline =====

class TestFullPipeline:
    """Test the complete scan → tokenize pipeline."""

    def test_scan_and_prepare_pipeline(self, tmp_path):
        """Create files, scan them, prepare training data, verify output."""
        # Setup directories
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        corpus_path = str(data_dir / "corpus.jsonl")
        state_path = str(data_dir / "watcher_state.json")
        training_data_path = str(data_dir / "training_data.bin")

        # Create sample files
        (watch_dir / "article.md").write_text(
            "# Machine Learning on Apple Silicon\n\n"
            "The Apple Neural Engine provides dedicated hardware for ML inference and training. "
            "With 16 cores on M3 Pro, it achieves 12.79 TFLOPS peak performance.\n\n"
            "## Training Results\n\n"
            "We successfully trained a 109M parameter model at 80.9ms per step.\n"
        )
        (watch_dir / "code.py").write_text(
            "import numpy as np\n\n"
            "def process_data(data):\n"
            "    \"\"\"Process input data for training.\"\"\"\n"
            "    normalized = (data - data.mean()) / data.std()\n"
            "    return normalized\n\n"
            "def train_step(model, batch):\n"
            "    loss = model.forward(batch)\n"
            "    model.backward(loss)\n"
            "    return loss\n"
        )
        (watch_dir / "config.json").write_text(json.dumps({
            "model": "stories110m",
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_steps": 1000,
        }, indent=2))

        # Step 1: Scan
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'collector'))
        import file_watcher

        from unittest.mock import patch
        with patch.object(file_watcher, 'DATA_DIR', str(data_dir)), \
             patch.object(file_watcher, 'CORPUS_FILE', corpus_path), \
             patch.object(file_watcher, 'STATE_FILE', state_path), \
             patch.object(file_watcher, 'CONFIG_FILE', '/nonexistent'):

            state = file_watcher.WatcherState(state_path)
            file_watcher.full_scan([str(watch_dir)], state)

        # Verify corpus
        assert os.path.exists(corpus_path)
        with open(corpus_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        assert len(entries) == 3

        collected_names = {os.path.basename(e["path"]) for e in entries}
        assert "article.md" in collected_names
        assert "code.py" in collected_names
        assert "config.json" in collected_names

        # Verify text content preserved
        article = next(e for e in entries if "article" in e["path"])
        assert "Apple Neural Engine" in article["text"]
        assert "12.79 TFLOPS" in article["text"]

        # Step 2: Tokenize
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tokenizer'))
        import prepare_training_data as tokenizer_mod

        docs = tokenizer_mod.load_corpus(corpus_path)
        assert len(docs) == 3

        # Use char tokenizer for deterministic test
        tok = tokenizer_mod.CharTokenizer()
        tok.build_vocab([d["text"] for d in docs])
        n_tokens = tokenizer_mod.tokenize_and_write(docs, tok, training_data_path)

        # Verify training data
        assert os.path.exists(training_data_path)
        assert n_tokens > 0

        file_size = os.path.getsize(training_data_path)
        assert file_size == n_tokens * 2  # uint16 = 2 bytes

        # Verify binary content is valid uint16
        with open(training_data_path, 'rb') as f:
            data = f.read()
        for i in range(0, min(len(data), 100), 2):
            val = struct.unpack('<H', data[i:i+2])[0]
            assert 0 <= val <= 65535

    def test_scan_skips_sensitive(self, tmp_path):
        """Verify sensitive files are never collected."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        corpus_path = str(data_dir / "corpus.jsonl")
        state_path = str(data_dir / "watcher_state.json")

        # Create both normal and sensitive files
        (watch_dir / "readme.md").write_text("This is a normal readme file with content.")
        (watch_dir / ".env").write_text("DATABASE_URL=postgres://user:pass@host/db")
        (watch_dir / "credentials.json").write_text('{"api_key": "sk-secret123"}')
        (watch_dir / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----\nfake key\n")

        ssh_dir = watch_dir / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "config").write_text("Host github.com\n  User git\n")

        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'collector'))
        import file_watcher

        from unittest.mock import patch
        with patch.object(file_watcher, 'DATA_DIR', str(data_dir)), \
             patch.object(file_watcher, 'CORPUS_FILE', corpus_path), \
             patch.object(file_watcher, 'STATE_FILE', state_path), \
             patch.object(file_watcher, 'CONFIG_FILE', '/nonexistent'):

            state = file_watcher.WatcherState(state_path)
            file_watcher.full_scan([str(watch_dir)], state)

        with open(corpus_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        collected_names = {os.path.basename(e["path"]) for e in entries}
        assert "readme.md" in collected_names
        assert ".env" not in collected_names
        assert "credentials.json" not in collected_names
        assert "id_rsa" not in collected_names

        # Verify no sensitive content leaked
        all_text = " ".join(e["text"] for e in entries)
        assert "sk-secret123" not in all_text
        assert "postgres://" not in all_text


# ===== Config Integration =====

class TestConfigIntegration:
    """Test that config.json is respected by the pipeline."""

    def test_config_controls_sources(self, tmp_path):
        """Config.json sources should control which dirs are scanned."""
        watch_dir1 = tmp_path / "enabled"
        watch_dir1.mkdir()
        (watch_dir1 / "included.txt").write_text("This file should be collected from enabled source.")

        watch_dir2 = tmp_path / "disabled"
        watch_dir2.mkdir()
        (watch_dir2 / "excluded.txt").write_text("This file should NOT be collected from disabled source.")

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "sources": [
                {"name": "Enabled", "path": str(watch_dir1), "enabled": True, "icon": "doc", "isCustom": False},
                {"name": "Disabled", "path": str(watch_dir2), "enabled": False, "icon": "doc", "isCustom": False},
            ],
            "fileTypes": {
                "plainText": True, "code": True, "richText": False,
                "pdf": False, "office": False, "email": False,
            }
        }
        config_path = str(data_dir / "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)

        corpus_path = str(data_dir / "corpus.jsonl")
        state_path = str(data_dir / "watcher_state.json")

        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'collector'))
        import file_watcher

        from unittest.mock import patch
        with patch.object(file_watcher, 'DATA_DIR', str(data_dir)), \
             patch.object(file_watcher, 'CORPUS_FILE', corpus_path), \
             patch.object(file_watcher, 'STATE_FILE', state_path), \
             patch.object(file_watcher, 'CONFIG_FILE', config_path):

            watch_dirs = file_watcher.get_watch_dirs_from_config()
            assert str(watch_dir1) in watch_dirs
            assert str(watch_dir2) not in watch_dirs

            state = file_watcher.WatcherState(state_path)
            file_watcher.full_scan(watch_dirs, state)

        with open(corpus_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        collected_names = {os.path.basename(e["path"]) for e in entries}
        assert "included.txt" in collected_names
        assert "excluded.txt" not in collected_names

    def test_config_controls_file_types(self, tmp_path):
        """Config file type settings should control which extensions are collected."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        (watch_dir / "readme.txt").write_text("A plain text file with enough content here.")
        (watch_dir / "main.py").write_text("def hello():\n    print('Hello World from Python')\n")

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Disable code, keep plain text
        config = {
            "sources": [
                {"name": "Test", "path": str(watch_dir), "enabled": True, "icon": "doc", "isCustom": False},
            ],
            "fileTypes": {
                "plainText": True, "code": False, "richText": False,
                "pdf": False, "office": False, "email": False,
            }
        }
        config_path = str(data_dir / "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)

        corpus_path = str(data_dir / "corpus.jsonl")
        state_path = str(data_dir / "watcher_state.json")

        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'collector'))
        import file_watcher

        from unittest.mock import patch
        with patch.object(file_watcher, 'DATA_DIR', str(data_dir)), \
             patch.object(file_watcher, 'CORPUS_FILE', corpus_path), \
             patch.object(file_watcher, 'STATE_FILE', state_path), \
             patch.object(file_watcher, 'CONFIG_FILE', config_path):

            text_exts, rich_exts = file_watcher.get_enabled_extensions()
            assert '.txt' in text_exts
            assert '.py' not in text_exts

            watch_dirs = file_watcher.get_watch_dirs_from_config()
            state = file_watcher.WatcherState(state_path)
            file_watcher.full_scan(watch_dirs, state)

        with open(corpus_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        collected_names = {os.path.basename(e["path"]) for e in entries}
        assert "readme.txt" in collected_names
        assert "main.py" not in collected_names


# ===== CLI Integration =====

class TestCLIIntegration:
    """Test the pai CLI script."""

    def test_pai_help(self):
        """pai help should exit 0 and show usage."""
        result = subprocess.run(
            [os.path.join(PROJECT_ROOT, 'pai'), 'help'],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "pai" in result.stdout.lower()
        assert "learn" in result.stdout
        assert "scan" in result.stdout
        assert "train" in result.stdout

    def test_pai_unknown_command(self):
        """Unknown command should exit non-zero."""
        result = subprocess.run(
            [os.path.join(PROJECT_ROOT, 'pai'), 'nonexistent_command'],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0
        assert "Unknown" in result.stdout or "unknown" in result.stderr.lower() or "Unknown" in result.stderr
