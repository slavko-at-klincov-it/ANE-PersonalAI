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
if [ ! -d "$DIR/repo" ]; then
    echo "Cloning ANE training repo..."
    git clone https://github.com/maderix/ANE.git "$DIR/repo"
else
    echo "  ANE repo: already present"
fi

# 3. Build trainer
echo "Building ANE trainer..."
cd "$DIR/repo/training/training_dynamic"
make MODEL=stories110m 2>&1 | tail -1
cd "$DIR"
echo "  Trainer: built"

# 4. Git LFS (for tokenizer.bin)
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    cd "$DIR/repo" && git lfs pull 2>/dev/null && cd "$DIR"
    echo "  Tokenizer: pulled via LFS"
else
    echo "  Note: install git-lfs for tokenizer (brew install git-lfs)"
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
echo "  pai train       # Train on ANE"
echo "  pai query       # Search your data"
