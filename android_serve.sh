#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${1:-8765}"
URL="http://127.0.0.1:${PORT}/visualized/sticker_dashboard.html"

echo "Serving dashboard at:"
echo "  ${URL}"
echo
echo "Open that URL in Chrome/Firefox on this phone."
echo "Press Ctrl+C in Termux to stop the server."

python -m http.server "${PORT}" --bind 127.0.0.1
