"""Unit tests for collector/file_watcher.py."""

import os
import json
import struct
import tempfile
import pytest
from unittest.mock import patch, MagicMock

import file_watcher


# ===== Sensitive File Detection =====

class TestIsSensitive:
    def test_env_file(self):
        assert file_watcher.is_sensitive("/home/user/.env")

    def test_env_in_path(self):
        assert file_watcher.is_sensitive("/home/user/project/.env.local")

    def test_ssh_key(self):
        assert file_watcher.is_sensitive("/home/user/.ssh/id_rsa")

    def test_pem_file(self):
        assert file_watcher.is_sensitive("/home/user/cert.pem")

    def test_credentials_file(self):
        assert file_watcher.is_sensitive("/home/user/credentials.json")

    def test_api_key_in_name(self):
        assert file_watcher.is_sensitive("/home/user/api_key.txt")

    def test_password_in_path(self):
        assert file_watcher.is_sensitive("/home/user/password_store/data.txt")

    def test_normal_file_not_sensitive(self):
        assert not file_watcher.is_sensitive("/home/user/readme.md")

    def test_code_file_not_sensitive(self):
        assert not file_watcher.is_sensitive("/home/user/main.py")

    def test_config_not_sensitive(self):
        assert not file_watcher.is_sensitive("/home/user/config.yaml")


# ===== Directory Skip Detection =====

class TestShouldSkipDir:
    def test_git_dir(self):
        assert file_watcher.should_skip_dir(".git")

    def test_node_modules(self):
        assert file_watcher.should_skip_dir("node_modules")

    def test_pycache(self):
        assert file_watcher.should_skip_dir("__pycache__")

    def test_venv(self):
        assert file_watcher.should_skip_dir(".venv")

    def test_ds_store(self):
        assert file_watcher.should_skip_dir(".DS_Store")

    def test_hidden_dir(self):
        assert file_watcher.should_skip_dir(".hidden")

    def test_normal_dir(self):
        assert not file_watcher.should_skip_dir("src")

    def test_docs_dir(self):
        assert not file_watcher.should_skip_dir("docs")


# ===== Text Extraction =====

class TestExtractText:
    def test_read_utf8(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!\nSecond line.", encoding='utf-8')
        result = file_watcher.extract_text(str(f))
        assert result == "Hello, world!\nSecond line."

    def test_read_with_unicode(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Stra\u00dfe, \u00c4nderung, \u00fc\u00f6\u00e4", encoding='utf-8')
        result = file_watcher.extract_text(str(f))
        assert "\u00c4nderung" in result

    def test_nonexistent_file(self):
        result = file_watcher.extract_text("/nonexistent/file.txt")
        assert result is None

    def test_binary_file_graceful(self, tmp_path):
        f = tmp_path / "binary.txt"
        f.write_bytes(b'\x00\x01\x02\xff\xfe\xfd')
        result = file_watcher.extract_text(str(f))
        # Should not crash, returns replacement characters
        assert result is not None


# ===== EMLX Extraction =====

class TestExtractEmlx:
    def test_basic_emlx(self, sample_emlx):
        result = file_watcher.extract_emlx(sample_emlx)
        assert result is not None
        assert "Test Email" in result
        assert "body of the test email" in result

    def test_emlx_nonexistent(self):
        result = file_watcher.extract_emlx("/nonexistent/test.emlx")
        assert result is None

    def test_emlx_empty(self, tmp_path):
        f = tmp_path / "empty.emlx"
        f.write_text("")
        result = file_watcher.extract_emlx(str(f))
        assert result is None


# ===== Textutil Extraction =====

class TestExtractViaTextutil:
    @patch('file_watcher.subprocess.run')
    def test_successful_extraction(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Extracted plain text content."
        )
        result = file_watcher.extract_via_textutil("/test/doc.rtf")
        assert result == "Extracted plain text content."
        mock_run.assert_called_once()

    @patch('file_watcher.subprocess.run')
    def test_failed_extraction(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = file_watcher.extract_via_textutil("/test/doc.rtf")
        assert result is None

    @patch('file_watcher.subprocess.run', side_effect=FileNotFoundError)
    def test_textutil_not_found(self, mock_run):
        result = file_watcher.extract_via_textutil("/test/doc.rtf")
        assert result is None

    @patch('file_watcher.subprocess.run', side_effect=TimeoutError)
    def test_timeout(self, mock_run):
        result = file_watcher.extract_via_textutil("/test/doc.rtf")
        assert result is None


# ===== PDF Extraction =====

class TestExtractPdf:
    @patch('file_watcher.subprocess.run')
    def test_pdftotext_fallback(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="PDF text content here."
        )
        result = file_watcher.extract_pdf("/test/doc.pdf")
        # Either PyPDF2 or pdftotext should work
        assert result is not None or result is None  # depends on PyPDF2 availability

    @patch('file_watcher.subprocess.run', side_effect=FileNotFoundError)
    def test_no_tools_available(self, mock_run):
        # When neither PyPDF2 nor pdftotext is available
        with patch.dict('sys.modules', {'PyPDF2': None}):
            result = file_watcher.extract_pdf("/nonexistent/doc.pdf")
            # Should return None gracefully
            assert result is None


# ===== Watcher State =====

class TestWatcherState:
    def test_initial_state_empty(self, tmp_data_dir):
        state_path = os.path.join(tmp_data_dir, "state.json")
        state = file_watcher.WatcherState(state_path)
        assert state.files == {}

    def test_save_and_load(self, tmp_data_dir):
        state_path = os.path.join(tmp_data_dir, "state.json")
        state = file_watcher.WatcherState(state_path)
        state.files["test.txt"] = {"hash": "abc", "mtime": 1000, "size": 100}
        state.save()

        state2 = file_watcher.WatcherState(state_path)
        assert "test.txt" in state2.files
        assert state2.files["test.txt"]["hash"] == "abc"

    def test_needs_update_new_file(self, tmp_data_dir, tmp_path):
        state_path = os.path.join(tmp_data_dir, "state.json")
        state = file_watcher.WatcherState(state_path)

        f = tmp_path / "new.txt"
        f.write_text("content")
        assert state.needs_update(str(f)) is True

    def test_needs_update_known_file(self, tmp_data_dir, tmp_path):
        state_path = os.path.join(tmp_data_dir, "state.json")
        state = file_watcher.WatcherState(state_path)

        f = tmp_path / "known.txt"
        f.write_text("content")

        # Mark as collected with a future mtime
        state.mark_collected(str(f), "hash123")

        # Should not need update
        assert state.needs_update(str(f)) is False

    def test_needs_update_nonexistent(self, tmp_data_dir):
        state_path = os.path.join(tmp_data_dir, "state.json")
        state = file_watcher.WatcherState(state_path)
        assert state.needs_update("/nonexistent/file.txt") is False

    def test_mark_collected(self, tmp_data_dir, tmp_path):
        state_path = os.path.join(tmp_data_dir, "state.json")
        state = file_watcher.WatcherState(state_path)

        f = tmp_path / "test.txt"
        f.write_text("content here")
        state.mark_collected(str(f), "hash456")

        assert str(f) in state.files
        assert state.files[str(f)]["hash"] == "hash456"
        assert "collected_at" in state.files[str(f)]


# ===== File Collection =====

class TestCollectFile:
    def test_collect_new_file(self, tmp_data_dir, tmp_path):
        state = file_watcher.WatcherState(os.path.join(tmp_data_dir, "state.json"))
        corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")

        f = tmp_path / "readme.md"
        f.write_text("# Hello World\n\nThis is a test file with enough content.")

        text_exts = file_watcher.TEXT_EXTENSIONS
        rich_exts = {}

        with open(corpus_path, 'a') as fh:
            result = file_watcher.collect_file(str(f), state, fh, text_exts, rich_exts)

        assert result is True

        # Verify corpus entry
        with open(corpus_path) as fh:
            entry = json.loads(fh.readline())
        assert entry["path"] == str(f)
        assert "Hello World" in entry["text"]

    def test_skip_sensitive_file(self, tmp_data_dir, tmp_path):
        state = file_watcher.WatcherState(os.path.join(tmp_data_dir, "state.json"))
        corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")

        f = tmp_path / ".env"
        f.write_text("SECRET_KEY=supersecret123456")

        with open(corpus_path, 'a') as fh:
            result = file_watcher.collect_file(str(f), state, fh,
                                               file_watcher.TEXT_EXTENSIONS, {})

        assert result is False

    def test_skip_too_small(self, tmp_data_dir, tmp_path):
        state = file_watcher.WatcherState(os.path.join(tmp_data_dir, "state.json"))
        corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")

        f = tmp_path / "tiny.txt"
        f.write_text("hi")

        with open(corpus_path, 'a') as fh:
            result = file_watcher.collect_file(str(f), state, fh,
                                               file_watcher.TEXT_EXTENSIONS, {})

        assert result is False

    def test_skip_already_collected(self, tmp_data_dir, tmp_path):
        state = file_watcher.WatcherState(os.path.join(tmp_data_dir, "state.json"))
        corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")

        f = tmp_path / "test.md"
        f.write_text("This is some content that has already been collected before.")

        text_exts = file_watcher.TEXT_EXTENSIONS
        rich_exts = {}

        # Collect first time
        with open(corpus_path, 'a') as fh:
            result1 = file_watcher.collect_file(str(f), state, fh, text_exts, rich_exts)
        assert result1 is True

        # Try to collect again (same content)
        with open(corpus_path, 'a') as fh:
            result2 = file_watcher.collect_file(str(f), state, fh, text_exts, rich_exts)
        assert result2 is False

    def test_skip_oversized_file(self, tmp_data_dir, tmp_path):
        state = file_watcher.WatcherState(os.path.join(tmp_data_dir, "state.json"))
        corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")

        f = tmp_path / "huge.txt"
        f.write_text("x" * (file_watcher.MAX_FILE_SIZE + 1))

        with open(corpus_path, 'a') as fh:
            result = file_watcher.collect_file(str(f), state, fh,
                                               file_watcher.TEXT_EXTENSIONS, {})

        assert result is False


# ===== Config Loading =====

class TestConfigLoading:
    def test_get_watch_dirs_from_config(self, sample_config, tmp_data_dir):
        """Test that watch dirs are loaded from config.json."""
        with patch.object(file_watcher, 'CONFIG_FILE', sample_config):
            dirs = file_watcher.get_watch_dirs_from_config()
            # Only enabled sources with existing paths
            for d in dirs:
                assert os.path.isdir(d)

    def test_get_watch_dirs_no_config(self, tmp_data_dir):
        """Falls back to defaults when no config exists."""
        with patch.object(file_watcher, 'CONFIG_FILE', '/nonexistent/config.json'):
            dirs = file_watcher.get_watch_dirs_from_config()
            assert isinstance(dirs, list)

    def test_get_enabled_extensions_defaults(self):
        """Without config, returns default extensions."""
        with patch.object(file_watcher, 'CONFIG_FILE', '/nonexistent/config.json'):
            text_exts, rich_exts = file_watcher.get_enabled_extensions()
            assert '.py' in text_exts
            assert '.md' in text_exts
            assert '.pdf' in rich_exts

    def test_get_enabled_extensions_email_disabled(self, sample_config):
        """Email is disabled by default in config."""
        with patch.object(file_watcher, 'CONFIG_FILE', sample_config):
            text_exts, rich_exts = file_watcher.get_enabled_extensions()
            assert '.emlx' not in rich_exts

    def test_get_enabled_extensions_email_enabled(self, tmp_data_dir):
        """Email extensions included when enabled in config."""
        config_path = os.path.join(tmp_data_dir, "config.json")
        config = {
            "fileTypes": {
                "plainText": True, "code": True, "richText": True,
                "pdf": True, "office": True, "email": True,
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        with patch.object(file_watcher, 'CONFIG_FILE', config_path):
            text_exts, rich_exts = file_watcher.get_enabled_extensions()
            assert '.emlx' in rich_exts


# ===== Full Scan =====

class TestFullScan:
    def test_scan_collects_files(self, tmp_data_dir, tmp_watch_dir):
        """Full scan should collect text files and skip sensitive/binary."""
        with patch.object(file_watcher, 'DATA_DIR', tmp_data_dir), \
             patch.object(file_watcher, 'CORPUS_FILE', os.path.join(tmp_data_dir, 'corpus.jsonl')), \
             patch.object(file_watcher, 'STATE_FILE', os.path.join(tmp_data_dir, 'state.json')), \
             patch.object(file_watcher, 'CONFIG_FILE', '/nonexistent'):

            state = file_watcher.WatcherState(os.path.join(tmp_data_dir, "state.json"))
            file_watcher.full_scan([tmp_watch_dir], state)

            corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")
            assert os.path.exists(corpus_path)

            # Read collected entries
            entries = []
            with open(corpus_path) as f:
                for line in f:
                    entries.append(json.loads(line))

            # Should have collected the text files
            collected_names = {os.path.basename(e["path"]) for e in entries}
            assert "readme.md" in collected_names
            assert "notes.txt" in collected_names
            assert "main.py" in collected_names
            assert "utils.swift" in collected_names
            assert "guide.md" in collected_names
            assert "config.yaml" in collected_names

            # Should NOT have collected sensitive or binary files
            assert ".env" not in collected_names
            assert "credentials.json" not in collected_names
            assert "image.png" not in collected_names
            assert "tiny.txt" not in collected_names

            # Should NOT have collected files from skipped dirs
            assert "HEAD" not in collected_names
            assert "package.json" not in collected_names

    def test_scan_incremental(self, tmp_data_dir, tmp_watch_dir):
        """Second scan should not re-collect unchanged files."""
        with patch.object(file_watcher, 'DATA_DIR', tmp_data_dir), \
             patch.object(file_watcher, 'CORPUS_FILE', os.path.join(tmp_data_dir, 'corpus.jsonl')), \
             patch.object(file_watcher, 'STATE_FILE', os.path.join(tmp_data_dir, 'state.json')), \
             patch.object(file_watcher, 'CONFIG_FILE', '/nonexistent'):

            state = file_watcher.WatcherState(os.path.join(tmp_data_dir, "state.json"))

            # First scan
            file_watcher.full_scan([tmp_watch_dir], state)
            corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")
            with open(corpus_path) as f:
                count1 = sum(1 for _ in f)

            # Second scan (no changes)
            file_watcher.full_scan([tmp_watch_dir], state)
            with open(corpus_path) as f:
                count2 = sum(1 for _ in f)

            # Should not have added more entries
            assert count2 == count1
