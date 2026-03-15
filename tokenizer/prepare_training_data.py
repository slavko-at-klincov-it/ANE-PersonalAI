#!/usr/bin/env python3
"""Convert collected corpus (JSONL) into pretokenized training data (uint16 binary).
Uses a simple BPE tokenizer (tiktoken/sentencepiece or character-level fallback).

Output format: flat uint16 token IDs — same as TinyStories/llama2.c format,
compatible with the ANE training pipeline.

Usage:
    python3 prepare_training_data.py
    python3 prepare_training_data.py --vocab-size 8192 --output ~/training_data.bin
"""

import os
import sys
import json
import struct
import argparse
from pathlib import Path

DATA_DIR = os.path.expanduser("~/.local/personal-ai")
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")

# Try to use tiktoken (OpenAI's fast BPE), fall back to character-level
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class CharTokenizer:
    """Simple character-level tokenizer as fallback.
    Maps each unique character to an ID. Vocab size = unique chars + special tokens.
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, texts, max_vocab=32000):
        """Build vocabulary from texts."""
        char_freq = {}
        for text in texts:
            for ch in text:
                char_freq[ch] = char_freq.get(ch, 0) + 1

        # Sort by frequency, take top max_vocab
        sorted_chars = sorted(char_freq.items(), key=lambda x: -x[1])
        self.char_to_id = {ch: i + 1 for i, (ch, _) in enumerate(sorted_chars[:max_vocab - 1])}
        self.char_to_id['<unk>'] = 0
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'char_to_id': self.char_to_id, 'vocab_size': self.vocab_size}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.char_to_id = data['char_to_id']
        self.vocab_size = data['vocab_size']
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}


class TiktokenWrapper:
    """Wrapper around tiktoken for consistent interface."""

    def __init__(self, encoding_name='cl100k_base'):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text):
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids):
        return self.enc.decode(ids)


def load_corpus(corpus_path):
    """Load all documents from JSONL corpus."""
    docs = []
    with open(corpus_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                text = entry.get('text', '')
                if text and len(text.strip()) > 10:
                    docs.append({
                        'text': text,
                        'path': entry.get('path', ''),
                        'size': len(text),
                    })
            except json.JSONDecodeError:
                continue
    return docs


def tokenize_and_write(docs, tokenizer, output_path, seq_len=256):
    """Tokenize all documents and write as flat uint16 binary."""
    all_tokens = []
    for doc in docs:
        tokens = tokenizer.encode(doc['text'])
        # Clamp to uint16 range
        tokens = [min(t, 65535) for t in tokens]
        all_tokens.extend(tokens)

    # Write as uint16
    n = len(all_tokens)
    with open(output_path, 'wb') as f:
        for t in all_tokens:
            f.write(struct.pack('<H', t))

    return n


def main():
    parser = argparse.ArgumentParser(description='Prepare training data from corpus')
    parser.add_argument('--corpus', default=CORPUS_FILE, help='Input JSONL corpus')
    parser.add_argument('--output', default=os.path.join(DATA_DIR, 'training_data.bin'),
                        help='Output binary file')
    parser.add_argument('--tokenizer', default='auto',
                        choices=['auto', 'tiktoken', 'char'],
                        help='Tokenizer to use')
    parser.add_argument('--vocab-size', type=int, default=32000,
                        help='Max vocab size for char tokenizer')
    parser.add_argument('--tokenizer-path', default=os.path.join(DATA_DIR, 'tokenizer.json'),
                        help='Path to save/load tokenizer state')
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"No corpus found at {args.corpus}")
        print("Run the file watcher first: python3 collector/file_watcher.py --scan")
        return 1

    # Load corpus
    docs = load_corpus(args.corpus)
    total_chars = sum(d['size'] for d in docs)
    print(f"Loaded {len(docs)} documents ({total_chars/1e6:.1f} MB text)")

    # Select tokenizer
    if args.tokenizer == 'auto':
        if HAS_TIKTOKEN:
            print("Using tiktoken (cl100k_base)")
            tokenizer = TiktokenWrapper()
        else:
            print("Using character-level tokenizer (install tiktoken for BPE)")
            tokenizer = CharTokenizer()
            texts = [d['text'] for d in docs]
            tokenizer.build_vocab(texts, args.vocab_size)
            tokenizer.save(args.tokenizer_path)
            print(f"  Vocab size: {tokenizer.vocab_size}")
    elif args.tokenizer == 'tiktoken':
        if not HAS_TIKTOKEN:
            print("tiktoken not installed: pip3 install tiktoken")
            return 1
        tokenizer = TiktokenWrapper()
    else:
        tokenizer = CharTokenizer()
        texts = [d['text'] for d in docs]
        tokenizer.build_vocab(texts, args.vocab_size)
        tokenizer.save(args.tokenizer_path)

    # Tokenize and write
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_tokens = tokenize_and_write(docs, tokenizer, args.output)
    size = os.path.getsize(args.output)
    print(f"Written {n_tokens} tokens ({size/1e6:.1f} MB) to {args.output}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Avg tokens/doc: {n_tokens/len(docs):.0f}")

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
