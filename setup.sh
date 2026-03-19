#!/bin/bash
# setup.sh — One-time setup for Personal AI
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== Personal AI Setup ==="

# 1. Python venv
echo "Setting up Python environment..."
if [ ! -d "$DIR/.venv" ]; then
    python3 -m venv "$DIR/.venv"
fi
source "$DIR/.venv/bin/activate"
pip install -q tiktoken watchdog
echo "  Python venv: OK (tiktoken + watchdog)"

# 2. ANE Training repo
ANE_DIR="$HOME/Code/ANE-Training"
if [ ! -d "$ANE_DIR" ]; then
    echo "Cloning ANE-Training repo..."
    git clone https://github.com/slavko-at-klincov-it/ANE-Training.git "$ANE_DIR"
else
    echo "  ANE-Training repo: already present at $ANE_DIR"
fi

# 3. Build trainer
echo "Building ANE trainer..."
cd "$ANE_DIR/training/training_dynamic"
make MODEL=stories110m 2>&1 | tail -1
cd "$DIR"
echo "  Trainer: built"

# 4. Git LFS (for checkpoint/tokenizer)
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    cd "$ANE_DIR" && git lfs pull 2>/dev/null && cd "$DIR"
    echo "  LFS data: pulled"
else
    echo "  Note: install git-lfs for large files (brew install git-lfs)"
fi

# 5. Data directory
mkdir -p ~/.local/personal-ai
echo "  Data dir: ~/.local/personal-ai"

# 6. Symlink pai to ~/bin
mkdir -p ~/bin
ln -sf "$DIR/pai" ~/bin/pai
echo "  CLI: ~/bin/pai (add ~/bin to PATH if needed)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  pai scan        # Collect your files"
echo "  pai tokenize    # Prepare training data"
echo "  pai learn       # Start continuous ANE training"
echo "  pai query       # Search your data"
