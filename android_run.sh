#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

python android_collect.py "$@"
python analyze.py
python visualize.py
python android_verify.py

echo
echo "Dashboard generated: visualized/sticker_dashboard.html"
echo "To view it from Android browser, run:"
echo "  bash android_serve.sh"
