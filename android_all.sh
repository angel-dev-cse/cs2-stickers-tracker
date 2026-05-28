#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${ANDROID_DASHBOARD_PORT:-8765}"
RUN_SETUP=1

if [ "${1:-}" = "--skip-setup" ]; then
  RUN_SETUP=0
  shift
fi

if [ "$RUN_SETUP" -eq 1 ]; then
  bash android_setup.sh
fi

bash android_run.sh "$@"

URL="http://127.0.0.1:${PORT}/visualized/sticker_dashboard.html"
LOG_FILE="${TMPDIR:-/data/data/com.termux/files/usr/tmp}/cs2_sticker_dashboard_server.log"
mkdir -p "$(dirname "${LOG_FILE}")"

echo
echo "Starting dashboard server:"
echo "  ${URL}"
echo

bash android_serve.sh "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID="$!"

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

sleep 1

if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
  echo "Server failed to start. Check: ${LOG_FILE}"
  exit 1
fi

if command -v termux-open-url >/dev/null 2>&1; then
  termux-open-url "${URL}" || true
elif command -v am >/dev/null 2>&1; then
  am start -a android.intent.action.VIEW -d "${URL}" >/dev/null 2>&1 || true
else
  echo "Open this URL in your mobile browser:"
  echo "  ${URL}"
fi

echo "Dashboard is running. Keep Termux open; press Ctrl+C to stop the server."
wait "${SERVER_PID}"
