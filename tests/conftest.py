"""Shared test fixtures for Personal AI tests."""

import os
import sys
import json
import shutil
import tempfile
import pytest

# Add project root to path so we can import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'collector'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tokenizer'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'inference'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'trainer'))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory mimicking ~/.local/personal-ai/."""
    data_dir = tmp_path / "personal-ai"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def tmp_watch_dir(tmp_path):
    """Temporary directory with sample files for scanning."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()

    # Plain text files
    (watch_dir / "readme.md").write_text("# My Project\n\nThis is a sample project with some content.")
    (watch_dir / "notes.txt").write_text("Meeting notes from today.\nDiscussed architecture changes.")
    (watch_dir / "config.yaml").write_text("server:\n  host: localhost\n  port: 8080\n")

    # Code files
    code_dir = watch_dir / "src"
    code_dir.mkdir()
    (code_dir / "main.py").write_text(
        "#!/usr/bin/env python3\n"
        "def hello():\n"
        "    print('Hello, world!')\n\n"
        "if __name__ == '__main__':\n"
        "    hello()\n"
    )
    (code_dir / "utils.swift").write_text(
        "import Foundation\n\n"
        "func greet(name: String) -> String {\n"
        "    return \"Hello, \\(name)!\"\n"
        "}\n"
    )

    # Subdirectory with more files
    sub_dir = watch_dir / "docs"
    sub_dir.mkdir()
    (sub_dir / "guide.md").write_text(
        "# User Guide\n\n"
        "## Getting Started\n\n"
        "Follow these steps to set up the project.\n\n"
        "## Configuration\n\n"
        "Edit config.yaml to customize settings.\n"
    )

    # Files that should be skipped
    git_dir = watch_dir / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    node_dir = watch_dir / "node_modules"
    node_dir.mkdir()
    (node_dir / "package.json").write_text('{"name": "dep"}')

    # Sensitive files that should be skipped
    (watch_dir / ".env").write_text("API_KEY=secret123\nDB_PASSWORD=hunter2")
    (watch_dir / "credentials.json").write_text('{"token": "abc123"}')

    # Binary/non-text file (should be ignored by extension)
    (watch_dir / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

    # Too-small file (should be skipped, < 10 chars)
    (watch_dir / "tiny.txt").write_text("hi")

    return str(watch_dir)


@pytest.fixture
def sample_corpus(tmp_data_dir):
    """Create a sample corpus.jsonl file."""
    corpus_path = os.path.join(tmp_data_dir, "corpus.jsonl")
    docs = [
        {
            "path": "/tmp/test/readme.md",
            "text": "# My Project\n\nThis is a sample project about machine learning.",
            "hash": "abc123",
            "collected_at": "2026-03-19T10:00:00",
            "size": 55,
        },
        {
            "path": "/tmp/test/notes.txt",
            "text": "Meeting notes: discussed ANE training performance improvements.",
            "hash": "def456",
            "collected_at": "2026-03-19T11:00:00",
            "size": 62,
        },
        {
            "path": "/tmp/test/main.py",
            "text": "def train_model():\n    print('Training on ANE...')\n    return True\n",
            "hash": "ghi789",
            "collected_at": "2026-03-19T12:00:00",
            "size": 68,
        },
    ]
    with open(corpus_path, 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')
    return corpus_path


@pytest.fixture
def sample_config(tmp_data_dir):
    """Create a sample config.json file."""
    config_path = os.path.join(tmp_data_dir, "config.json")
    config = {
        "sources": [
            {"name": "Docs", "path": "~/Documents", "enabled": True, "icon": "doc", "isCustom": False},
            {"name": "Code", "path": "~/Code", "enabled": True, "icon": "code", "isCustom": False},
            {"name": "Mail", "path": "~/Library/Mail", "enabled": False, "icon": "mail", "isCustom": False},
        ],
        "fileTypes": {
            "plainText": True,
            "code": True,
            "richText": True,
            "pdf": True,
            "office": True,
            "email": False,
        },
        "training": {
            "debounceSeconds": 30,
            "minSteps": 10,
            "maxSteps": 50,
            "nightlyEnabled": False,
        },
        "firstLaunchComplete": True,
        "paiPath": "~/bin/pai",
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def sample_emlx(tmp_path):
    """Create a sample Apple Mail .emlx file."""
    emlx_path = tmp_path / "test.emlx"
    email_content = (
        "From: sender@example.com\r\n"
        "To: receiver@example.com\r\n"
        "Subject: Test Email\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "This is the body of the test email.\r\n"
        "It has multiple lines.\r\n"
    )
    byte_count = len(email_content.encode('utf-8'))
    emlx_content = f"{byte_count}\n{email_content}"
    emlx_path.write_text(emlx_content)
    return str(emlx_path)
