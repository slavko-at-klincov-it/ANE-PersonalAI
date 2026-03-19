"""Unit tests for tokenizer/prepare_training_data.py."""

import os
import json
import struct
import pytest

import prepare_training_data as tokenizer


# ===== Character Tokenizer =====

class TestCharTokenizer:
    def test_build_vocab(self):
        t = tokenizer.CharTokenizer()
        t.build_vocab(["hello world", "hello"], max_vocab=100)
        assert t.vocab_size > 0
        assert t.vocab_size <= 100
        # 'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd' + <unk>
        assert t.vocab_size >= 9

    def test_encode_decode_roundtrip(self):
        t = tokenizer.CharTokenizer()
        t.build_vocab(["hello world"])
        encoded = t.encode("hello")
        decoded = t.decode(encoded)
        assert decoded == "hello"

    def test_unknown_char(self):
        t = tokenizer.CharTokenizer()
        t.build_vocab(["abc"])
        encoded = t.encode("xyz")
        # Unknown chars should map to 0 (<unk>)
        assert all(c == 0 for c in encoded)

    def test_empty_text(self):
        t = tokenizer.CharTokenizer()
        t.build_vocab(["hello"])
        assert t.encode("") == []
        assert t.decode([]) == ""

    def test_save_load(self, tmp_path):
        t = tokenizer.CharTokenizer()
        t.build_vocab(["the quick brown fox"])

        path = str(tmp_path / "tokenizer.json")
        t.save(path)

        t2 = tokenizer.CharTokenizer()
        t2.load(path)
        assert t2.vocab_size == t.vocab_size
        assert t2.encode("fox") == t.encode("fox")

    def test_max_vocab_limit(self):
        t = tokenizer.CharTokenizer()
        # Text with many unique chars
        text = "".join(chr(i) for i in range(32, 200))
        t.build_vocab([text], max_vocab=10)
        assert t.vocab_size <= 10


# ===== Tiktoken Wrapper =====

class TestTiktokenWrapper:
    @pytest.fixture
    def tiktoken_available(self):
        pytest.importorskip("tiktoken")

    def test_encode_decode_roundtrip(self, tiktoken_available):
        t = tokenizer.TiktokenWrapper()
        text = "Hello, world! This is a test."
        encoded = t.encode(text)
        decoded = t.decode(encoded)
        assert decoded == text

    def test_vocab_size(self, tiktoken_available):
        t = tokenizer.TiktokenWrapper()
        assert t.vocab_size > 50000  # cl100k_base has 100K+ tokens

    def test_encode_empty(self, tiktoken_available):
        t = tokenizer.TiktokenWrapper()
        assert t.encode("") == []

    def test_encode_unicode(self, tiktoken_available):
        t = tokenizer.TiktokenWrapper()
        encoded = t.encode("Stra\u00dfe \u00c4nderung \u00fc\u00f6\u00e4")
        assert len(encoded) > 0
        decoded = t.decode(encoded)
        assert "\u00c4nderung" in decoded


# ===== Corpus Loading =====

class TestLoadCorpus:
    def test_load_valid_corpus(self, sample_corpus):
        docs = tokenizer.load_corpus(sample_corpus)
        assert len(docs) == 3
        assert docs[0]["path"] == "/tmp/test/readme.md"
        assert "machine learning" in docs[0]["text"]

    def test_load_empty_corpus(self, tmp_path):
        corpus = tmp_path / "empty.jsonl"
        corpus.write_text("")
        docs = tokenizer.load_corpus(str(corpus))
        assert docs == []

    def test_load_with_invalid_lines(self, tmp_path):
        corpus = tmp_path / "mixed.jsonl"
        corpus.write_text(
            '{"path":"a.txt","text":"valid content here enough text","size":30}\n'
            'not json\n'
            '{"path":"b.txt","text":"another valid doc with content","size":30}\n'
        )
        docs = tokenizer.load_corpus(str(corpus))
        assert len(docs) == 2

    def test_load_skips_tiny_docs(self, tmp_path):
        corpus = tmp_path / "tiny.jsonl"
        corpus.write_text(
            '{"path":"a.txt","text":"hi","size":2}\n'
            '{"path":"b.txt","text":"This is long enough content to pass","size":35}\n'
        )
        docs = tokenizer.load_corpus(str(corpus))
        assert len(docs) == 1


# ===== Tokenize and Write =====

class TestTokenizeAndWrite:
    def test_writes_uint16_binary(self, tmp_path):
        docs = [{"text": "hello world"}]
        t = tokenizer.CharTokenizer()
        t.build_vocab(["hello world"])

        output = str(tmp_path / "training_data.bin")
        n = tokenizer.tokenize_and_write(docs, t, output)

        assert n == len("hello world")  # char-level: 1 token per char
        assert os.path.exists(output)

        # Verify binary format (uint16 little-endian)
        with open(output, 'rb') as f:
            data = f.read()
        assert len(data) == n * 2  # 2 bytes per uint16

        # Verify values are valid uint16
        for i in range(0, len(data), 2):
            val = struct.unpack('<H', data[i:i+2])[0]
            assert 0 <= val <= 65535

    def test_clamps_to_uint16(self, tmp_path):
        """Tokens > 65535 should be clamped."""
        docs = [{"text": "test"}]

        class FakeTokenizer:
            def encode(self, text):
                return [70000, 1, 2, 3]  # 70000 > uint16 max

        output = str(tmp_path / "data.bin")
        n = tokenizer.tokenize_and_write(docs, FakeTokenizer(), output)
        assert n == 4

        with open(output, 'rb') as f:
            val = struct.unpack('<H', f.read(2))[0]
        assert val == 65535  # clamped

    def test_empty_docs(self, tmp_path):
        output = str(tmp_path / "empty.bin")
        n = tokenizer.tokenize_and_write([], tokenizer.CharTokenizer(), output)
        assert n == 0

    def test_multiple_docs(self, tmp_path):
        docs = [{"text": "aaa"}, {"text": "bbb"}]
        t = tokenizer.CharTokenizer()
        t.build_vocab(["aaa bbb"])

        output = str(tmp_path / "multi.bin")
        n = tokenizer.tokenize_and_write(docs, t, output)
        assert n == 6  # 3 + 3 chars
