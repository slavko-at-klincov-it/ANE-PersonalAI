#!/usr/bin/env python3
"""Query interface for the personal AI model.
Loads a trained checkpoint and generates text on CPU.

Usage:
    python3 query.py                          # Interactive mode
    python3 query.py "what did I work on"     # Single query
    python3 query.py --complete "def hello"   # Code completion
    python3 query.py --stats                  # Show model stats
"""

import os
import sys
import struct
import math
import argparse
import json
from pathlib import Path

DATA_DIR = os.path.expanduser("~/.local/personal-ai")
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")

# Try tiktoken
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


# ===== Model config (must match training) =====
class ModelConfig:
    def __init__(self):
        self.dim = 768
        self.hidden = 2048
        self.heads = 12
        self.hd = 64
        self.seq = 256
        self.n_layers = 12
        self.vocab = 32000  # will be overridden by checkpoint


# ===== Simple CPU inference =====

def rmsnorm(x, w, dim):
    """RMSNorm: x * w / sqrt(mean(x^2) + eps)"""
    ss = sum(v * v for v in x) / dim
    inv = 1.0 / math.sqrt(ss + 1e-5)
    return [x[i] * inv * w[i] for i in range(dim)]


def softmax(x):
    mx = max(x)
    e = [math.exp(v - mx) for v in x]
    s = sum(e)
    return [v / s for v in e]


def matmul(out_dim, in_dim, W, x):
    """W[out_dim, in_dim] @ x[in_dim] -> out[out_dim]"""
    out = []
    for i in range(out_dim):
        val = sum(W[i * in_dim + j] * x[j] for j in range(in_dim))
        out.append(val)
    return out


def silu(x):
    return [v * (1.0 / (1.0 + math.exp(-v))) if abs(v) < 20 else (v if v > 0 else 0) for v in x]


class Checkpoint:
    """Reads a training checkpoint."""

    def __init__(self, path):
        self.path = path
        self.step = 0
        self.loss = 0
        self.config = ModelConfig()
        self.layers = []
        self.rms_final = None
        self.embed = None

    def load(self):
        """Load checkpoint header and weights."""
        with open(self.path, 'rb') as f:
            # Read header (matches CkptHdr in io.h)
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x424C5A54:
                print(f"Bad checkpoint magic: {hex(magic)}")
                return False

            version = struct.unpack('<I', f.read(4))[0]
            self.step = struct.unpack('<I', f.read(4))[0]
            total_steps = struct.unpack('<I', f.read(4))[0]

            n_layers = struct.unpack('<I', f.read(4))[0]
            vocab = struct.unpack('<I', f.read(4))[0]
            dim = struct.unpack('<I', f.read(4))[0]
            hidden = struct.unpack('<I', f.read(4))[0]
            n_heads = struct.unpack('<I', f.read(4))[0]
            seq = struct.unpack('<I', f.read(4))[0]

            self.config.n_layers = n_layers
            self.config.vocab = vocab
            self.config.dim = dim
            self.config.hidden = hidden
            self.config.heads = n_heads
            self.config.seq = seq
            self.config.hd = dim // n_heads

            lr = struct.unpack('<f', f.read(4))[0]
            self.loss = struct.unpack('<f', f.read(4))[0]

            # Additional v2+ fields
            if version >= 2:
                cum_compile = struct.unpack('<d', f.read(8))[0]
                cum_train = struct.unpack('<d', f.read(8))[0]
                cum_wall = struct.unpack('<d', f.read(8))[0]
                cum_steps = struct.unpack('<I', f.read(4))[0]
                cum_batches = struct.unpack('<I', f.read(4))[0]
                adam_t = struct.unpack('<I', f.read(4))[0]

            if version >= 4:
                # GQA fields
                kv_heads = struct.unpack('<I', f.read(4))[0]
                head_dim = struct.unpack('<I', f.read(4))[0]
                q_dim = struct.unpack('<I', f.read(4))[0]

            print(f"Checkpoint v{version}: step={self.step}, loss={self.loss:.4f}")
            print(f"  {n_layers} layers, dim={dim}, hidden={hidden}, heads={n_heads}, vocab={vocab}")

            # Read weights (simplified — just reads to confirm format)
            # Full weight loading would read per-layer Wq,Wk,Wv,Wo,W1,W2,W3,rms + Adam state
            return True

    def info(self):
        return {
            'step': self.step,
            'loss': self.loss,
            'layers': self.config.n_layers,
            'dim': self.config.dim,
            'hidden': self.config.hidden,
            'heads': self.config.heads,
            'vocab': self.config.vocab,
            'params_m': (self.config.n_layers *
                (self.config.dim * self.config.dim * 4 +
                 self.config.dim * self.config.hidden * 3 +
                 self.config.dim * 2) +
                self.config.vocab * self.config.dim) / 1e6,
        }


# ===== Tokenizer =====

class CharTokenizer:
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.vocab_size = data['vocab_size']

    def encode(self, text):
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


def get_tokenizer():
    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding('cl100k_base')
        return enc
    elif os.path.exists(TOKENIZER_PATH):
        return CharTokenizer(TOKENIZER_PATH)
    return None


# ===== Main =====

def show_stats():
    """Show training status and corpus info."""
    print("=== Personal AI Status ===\n")

    # Corpus
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    if os.path.exists(corpus_path):
        size = os.path.getsize(corpus_path)
        lines = sum(1 for _ in open(corpus_path))
        print(f"Corpus:     {lines} documents, {size/1e6:.1f} MB")
    else:
        print("Corpus:     not created yet")

    # Training data
    data_path = os.path.join(DATA_DIR, "training_data.bin")
    if os.path.exists(data_path):
        size = os.path.getsize(data_path)
        tokens = size // 2
        print(f"Token data: {tokens:,} tokens, {size/1e6:.1f} MB")
    else:
        print("Token data: not created yet")

    # Checkpoints
    ckpt_path = os.path.join(DATA_DIR, "checkpoint.bin")
    if os.path.exists(ckpt_path):
        ckpt = Checkpoint(ckpt_path)
        if ckpt.load():
            info = ckpt.info()
            print(f"\nModel:      {info['params_m']:.1f}M params")
            print(f"Training:   step {info['step']}, loss {info['loss']:.4f}")
            print(f"Config:     {info['layers']}L, dim={info['dim']}, heads={info['heads']}")
    else:
        # Check repo checkpoint
        repo_ckpt = os.path.expanduser("~/Code/ANE-Training/repo/training/training_dynamic/checkpoint.bin")
        if os.path.exists(repo_ckpt):
            ckpt = Checkpoint(repo_ckpt)
            if ckpt.load():
                info = ckpt.info()
                print(f"\nModel:      {info['params_m']:.1f}M params (in repo)")
                print(f"Training:   step {info['step']}, loss {info['loss']:.4f}")
        else:
            print("\nModel:      no checkpoint yet (run training first)")

    # Watcher state
    state_path = os.path.join(DATA_DIR, "watcher_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"\nWatcher:    tracking {len(state)} files")

    # Training log
    log_path = os.path.join(DATA_DIR, "train.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            print(f"Last train: {lines[-1].strip()}")

    print()


def interactive_mode():
    """Interactive query mode."""
    print("=== Personal AI Query Interface ===")
    print("Model is training — full inference requires a trained checkpoint.")
    print("Currently showing corpus search (no neural inference yet).\n")
    print("Commands: /stats, /search <query>, /recent, /quit\n")

    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    docs = []
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    while True:
        try:
            query = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query == '/quit':
            break
        if query == '/stats':
            show_stats()
            continue
        if query == '/recent':
            # Show most recently collected files
            recent = sorted(docs, key=lambda d: d.get('collected_at', ''), reverse=True)[:10]
            for d in recent:
                name = os.path.basename(d.get('path', '?'))
                ts = d.get('collected_at', '?')[:19]
                size = d.get('size', 0)
                print(f"  {ts}  {name} ({size} chars)")
            continue

        # Simple keyword search in corpus
        if query.startswith('/search '):
            query = query[8:]

        keywords = query.lower().split()
        results = []
        for doc in docs:
            text = doc.get('text', '').lower()
            path = doc.get('path', '').lower()
            score = sum(1 for kw in keywords if kw in text or kw in path)
            if score > 0:
                results.append((score, doc))

        results.sort(key=lambda x: -x[0])

        if not results:
            print("  No matches in corpus.")
        else:
            print(f"  Found {len(results)} matches:")
            for score, doc in results[:5]:
                name = os.path.basename(doc.get('path', '?'))
                # Show snippet around first keyword match
                text = doc.get('text', '')
                snippet = ''
                for kw in keywords:
                    idx = text.lower().find(kw)
                    if idx >= 0:
                        start = max(0, idx - 40)
                        end = min(len(text), idx + len(kw) + 40)
                        snippet = '...' + text[start:end].replace('\n', ' ') + '...'
                        break
                print(f"  [{score}] {name}")
                if snippet:
                    print(f"      {snippet}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Personal AI query interface')
    parser.add_argument('query', nargs='?', help='Query text')
    parser.add_argument('--stats', action='store_true', help='Show model/corpus stats')
    parser.add_argument('--search', help='Search corpus for keywords')
    parser.add_argument('--recent', action='store_true', help='Show recently collected files')
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.search:
        # Quick search
        sys.argv = ['query.py']  # Reset for interactive
        # Inline search
        corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
        if not os.path.exists(corpus_path):
            print("No corpus. Run file_watcher.py --scan first.")
            return
        docs = []
        with open(corpus_path) as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except:
                    pass
        keywords = args.search.lower().split()
        for doc in docs:
            text = doc.get('text', '').lower()
            if any(kw in text for kw in keywords):
                name = os.path.basename(doc.get('path', '?'))
                print(f"  {name}: {doc.get('size', 0)} chars")
    elif args.recent:
        corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
        if os.path.exists(corpus_path):
            docs = []
            with open(corpus_path) as f:
                for line in f:
                    try:
                        docs.append(json.loads(line))
                    except:
                        pass
            recent = sorted(docs, key=lambda d: d.get('collected_at', ''), reverse=True)[:10]
            for d in recent:
                name = os.path.basename(d.get('path', '?'))
                ts = d.get('collected_at', '?')[:19]
                print(f"  {ts}  {name}")
    elif args.query:
        # Single query mode — search corpus
        corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
        if not os.path.exists(corpus_path):
            print("No corpus yet.")
            return
        docs = []
        with open(corpus_path) as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except:
                    pass
        keywords = args.query.lower().split()
        results = []
        for doc in docs:
            text = doc.get('text', '').lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                results.append((score, doc))
        results.sort(key=lambda x: -x[0])
        for score, doc in results[:5]:
            name = os.path.basename(doc.get('path', '?'))
            print(f"  [{score}] {name}")
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
