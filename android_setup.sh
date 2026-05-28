#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

pkg update -y
pkg install -y python git ca-certificates openssl

if ! pkg install -y python-numpy python-pandas; then
  echo "Termux pandas/numpy packages were unavailable. Falling back to pip; this can be slow."
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install --prefer-binary numpy pandas
fi

python -m pip install --upgrade pip
python -m pip install -r android_requirements.txt

mkdir -p data/snapshots data/history analyze visualized

echo
echo "Android setup complete."
echo "Run: bash android_run.sh"
