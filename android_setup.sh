#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

pkg update -y
pkg install -y python git ca-certificates openssl

if ! python - <<'PY' >/dev/null 2>&1
import numpy
import pandas
PY
then
  pkg install -y python-numpy
  if ! pkg install -y python-pandas; then
    echo "python-pandas was not found in the default Termux repo. Trying TUR..."
    pkg install -y tur-repo || true
    pkg update -y
    pkg install -y python-pandas
  fi
fi

python - <<'PY'
import numpy
import pandas
print("numpy/pandas OK")
PY

mkdir -p data/snapshots data/history analyze visualized

echo
echo "Android setup complete."
echo "Run: bash android_run.sh"
