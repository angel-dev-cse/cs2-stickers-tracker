#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

COLLECT_ARGS=("$@")
ANALYZE_ARGS=(--history-source latest)
VISUALIZE_ARGS=(--history-source latest --no-fetch-2p)

if [ "${ANDROID_FULL_HISTORY:-0}" != "1" ]; then
  COLLECT_ARGS+=(--no-cumulative-history)
else
  ANALYZE_ARGS=(--history-source full)
  VISUALIZE_ARGS=(--history-source full --no-fetch-2p)
fi

python android_collect.py "${COLLECT_ARGS[@]}"
python analyze.py "${ANALYZE_ARGS[@]}"
python visualize.py "${VISUALIZE_ARGS[@]}"
python android_verify.py

echo
echo "Dashboard generated: visualized/sticker_dashboard.html"
echo "To view it from Android browser, run:"
echo "  bash android_serve.sh"
