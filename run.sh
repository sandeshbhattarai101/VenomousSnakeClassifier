#!/usr/bin/env bash
set -e

# ── Colour helpers ───────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo -e "${CYAN}╔══════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║       Snake Vision — Startup Script      ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${RESET}"

# ── Python / venv ────────────────────────────────────────────────
PYTHON=python3
if [ -d "venv" ]; then
  source venv/bin/activate
  PYTHON=python
fi

# ── Check model exists ───────────────────────────────────────────
if [ ! -f "saved_models/snake_classifier.pth" ]; then
  echo -e "${YELLOW}⚠  No trained model found.${RESET}"
  echo ""
  echo "Run the following steps first:"
  echo ""
  echo "  1. Install dependencies:"
  echo "       pip install -r requirements.txt"
  echo ""
  echo "  2. Download training data:"
  echo "       python download_data.py"
  echo ""
  echo "  3. Train the model (takes ~30–90 min depending on hardware):"
  echo "       python train.py"
  echo ""
  echo "  4. (Optional) Evaluate:"
  echo "       python evaluate.py"
  echo ""
  echo "Then re-run:  bash run.sh"
  exit 1
fi

echo -e "${GREEN}✓ Model found.${RESET} Starting server…"
echo ""
echo -e "  ${CYAN}→ Open in your browser: http://localhost:8000${RESET}"
echo -e "  ${CYAN}→ On your phone (same Wi-Fi): http://$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}'):8000${RESET}"
echo ""
echo "Press Ctrl+C to stop."
echo ""

$PYTHON app.py
