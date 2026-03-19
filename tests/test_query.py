"""Unit tests for inference/query.py."""

import os
import json
import struct
import pytest
from unittest.mock import patch
from io import StringIO

import query


# ===== ModelConfig =====

class TestModelConfig:
    def test_default_config(self):
        config = query.ModelConfig()
        assert config.dim == 768
        assert config.n_layers == 12
        assert config.heads == 12
        assert config.seq == 256
        assert config.vocab == 32000


# ===== Checkpoint =====

class TestCheckpoint:
    def test_invalid_file(self, tmp_path):
        f = tmp_path / "bad.bin"
        f.write_bytes(b'\x00\x00\x00\x00')
        ckpt = query.Checkpoint(str(f))
        assert ckpt.load() is False

    def test_valid_header(self, tmp_path):
        """Create a minimal valid checkpoint header."""
        f = tmp_path / "checkpoint.bin"
        with open(str(f), 'wb') as fh:
            fh.write(struct.pack('<I', 0x424C5A54))  # magic
            fh.write(struct.pack('<I', 1))   # version
            fh.write(struct.pack('<I', 50))  # step
            fh.write(struct.pack('<I', 100)) # total_steps
            fh.write(struct.pack('<I', 12))  # n_layers
            fh.write(struct.pack('<I', 32000))  # vocab
            fh.write(struct.pack('<I', 768))    # dim
            fh.write(struct.pack('<I', 2048))   # hidden
            fh.write(struct.pack('<I', 12))     # n_heads
            fh.write(struct.pack('<I', 256))    # seq
            fh.write(struct.pack('<f', 0.001))  # lr
            fh.write(struct.pack('<f', 2.5))    # loss

        ckpt = query.Checkpoint(str(f))
        result = ckpt.load()
        assert result is True
        assert ckpt.step == 50
        assert ckpt.loss == pytest.approx(2.5, abs=0.01)
        assert ckpt.config.dim == 768

    def test_checkpoint_info(self, tmp_path):
        f = tmp_path / "checkpoint.bin"
        with open(str(f), 'wb') as fh:
            fh.write(struct.pack('<I', 0x424C5A54))
            fh.write(struct.pack('<I', 1))
            fh.write(struct.pack('<I', 100))
            fh.write(struct.pack('<I', 200))
            fh.write(struct.pack('<I', 12))
            fh.write(struct.pack('<I', 32000))
            fh.write(struct.pack('<I', 768))
            fh.write(struct.pack('<I', 2048))
            fh.write(struct.pack('<I', 12))
            fh.write(struct.pack('<I', 256))
            fh.write(struct.pack('<f', 0.001))
            fh.write(struct.pack('<f', 1.8))

        ckpt = query.Checkpoint(str(f))
        ckpt.load()
        info = ckpt.info()
        assert info['step'] == 100
        assert info['layers'] == 12
        assert info['dim'] == 768
        assert info['params_m'] > 0


# ===== Math Functions =====

class TestMathFunctions:
    def test_softmax_uniform(self):
        result = query.softmax([0, 0, 0])
        assert len(result) == 3
        assert pytest.approx(sum(result), abs=1e-6) == 1.0
        assert pytest.approx(result[0], abs=1e-6) == 1/3

    def test_softmax_peaked(self):
        result = query.softmax([10, 0, 0])
        assert result[0] > 0.99
        assert sum(result) == pytest.approx(1.0, abs=1e-6)

    def test_rmsnorm(self):
        x = [1.0, 2.0, 3.0]
        w = [1.0, 1.0, 1.0]
        result = query.rmsnorm(x, w, 3)
        assert len(result) == 3
        # Should be normalized
        assert all(abs(v) < 10 for v in result)

    def test_silu(self):
        result = query.silu([0.0, 1.0, -1.0])
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[1] > 0  # silu(1) > 0
        assert result[2] < 0  # silu(-1) < 0

    def test_silu_large_values(self):
        result = query.silu([100.0, -100.0])
        assert result[0] == pytest.approx(100.0, abs=0.1)
        assert result[1] == 0

    def test_matmul(self):
        # 2x3 matrix times 3-vector
        W = [1, 0, 0,   # row 1
             0, 1, 0]   # row 2
        x = [5, 7, 11]
        result = query.matmul(2, 3, W, x)
        assert result == [5, 7]


# ===== Tokenizer =====

class TestGetTokenizer:
    def test_tiktoken_if_available(self):
        if query.HAS_TIKTOKEN:
            tok = query.get_tokenizer()
            assert tok is not None

    def test_char_tokenizer_fallback(self, tmp_path):
        tok_path = str(tmp_path / "tokenizer.json")
        data = {"char_to_id": {"a": 1, "b": 2, "<unk>": 0}, "vocab_size": 3}
        with open(tok_path, 'w') as f:
            json.dump(data, f)

        with patch.object(query, 'HAS_TIKTOKEN', False), \
             patch.object(query, 'TOKENIZER_PATH', tok_path):
            tok = query.get_tokenizer()
            assert tok is not None
            assert tok.encode("ab") == [1, 2]


# ===== Stats Display =====

class TestShowStats:
    def test_no_data(self, tmp_data_dir, capsys):
        with patch.object(query, 'DATA_DIR', tmp_data_dir):
            query.show_stats()
            output = capsys.readouterr().out
            assert "Personal AI Status" in output
            assert "not created" in output

    def test_with_corpus(self, sample_corpus, tmp_data_dir, capsys):
        with patch.object(query, 'DATA_DIR', tmp_data_dir):
            query.show_stats()
            output = capsys.readouterr().out
            assert "3 documents" in output

    def test_with_training_data(self, tmp_data_dir, sample_corpus, capsys):
        # Create training data
        data_path = os.path.join(tmp_data_dir, "training_data.bin")
        with open(data_path, 'wb') as f:
            for i in range(1000):
                f.write(struct.pack('<H', i % 100))

        with patch.object(query, 'DATA_DIR', tmp_data_dir):
            query.show_stats()
            output = capsys.readouterr().out
            assert "1,000 tokens" in output or "1.000 tokens" in output


# ===== Keyword Search =====

class TestKeywordSearch:
    """Tests for the corpus search functionality."""

    def test_search_single_keyword(self, sample_corpus):
        """Load corpus and search for a keyword."""
        docs = []
        with open(sample_corpus) as f:
            for line in f:
                docs.append(json.loads(line))

        keywords = ["machine"]
        results = []
        for doc in docs:
            text = doc.get('text', '').lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                results.append((score, doc))

        assert len(results) == 1
        assert "readme" in results[0][1]["path"]

    def test_search_multiple_keywords(self, sample_corpus):
        docs = []
        with open(sample_corpus) as f:
            for line in f:
                docs.append(json.loads(line))

        keywords = ["ane", "training"]
        results = []
        for doc in docs:
            text = doc.get('text', '').lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                results.append((score, doc))
        results.sort(key=lambda x: -x[0])

        # "notes.txt" and "main.py" mention ANE/training
        assert len(results) >= 1

    def test_search_no_results(self, sample_corpus):
        docs = []
        with open(sample_corpus) as f:
            for line in f:
                docs.append(json.loads(line))

        keywords = ["nonexistentkeyword"]
        results = []
        for doc in docs:
            text = doc.get('text', '').lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                results.append((score, doc))

        assert len(results) == 0
