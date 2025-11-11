#!/usr/bin/env bash
# Bootstrap env for edge-vision-pipeline on Raspberry Pi 5
# Usage: ./scripts/setup_pi_env.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME=".venv"
PYTHON_BIN="python3"

echo "📁 Project root: $PROJECT_ROOT"

echo "🔧 Installing system packages (requires sudo)…"
sudo apt update
sudo apt install -y \
  python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  gstreamer1.0-gl ffmpeg

cd "$PROJECT_ROOT"
if [[ ! -d "$VENV_NAME" ]]; then
  echo "🌀 Creating virtual environment: $VENV_NAME"
  $PYTHON_BIN -m venv "$VENV_NAME"
else
  echo "✅ Virtual environment already exists: $VENV_NAME"
fi

# shellcheck disable=SC1090
source "$VENV_NAME/bin/activate"
echo "✅ Activated venv: $(python -V)"

pip install --upgrade pip
pip install -r requirements.txt

# Keep project on PYTHONPATH when venv activates
ACTIVATE_FILE="$PROJECT_ROOT/$VENV_NAME/bin/activate"
if ! grep -q "PYTHONPATH" "$ACTIVATE_FILE"; then
  echo "export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"" >> "$ACTIVATE_FILE"
  echo "🔗 Added PROJECT_ROOT to PYTHONPATH in venv activate"
fi

echo
echo "🎯 Setup complete."
echo "Next:"
echo "  source $PROJECT_ROOT/$VENV_NAME/bin/activate"
echo "  python -c \"import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; print(Gst.version_string())\""
