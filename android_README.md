# Android Standalone Setup

This project can run standalone on Android through Termux. The Android path avoids Playwright/Chromium and uses `android_collect.py`, which reads CS2Tokens paginated HTML directly with Python's standard library, then fetches each sticker's metadata/history JSON.

## Files

- `android_collect.py` - phone-friendly collector, no browser required.
- `android_setup.sh` - installs Termux packages and Python requirements.
- `android_run.sh` - runs collect, analyze, and visualize.
- `android_serve.sh` - serves the dashboard to your Android browser.
- `android_requirements.txt` - notes only; Android setup uses Termux packages, not global pip.
- `android_verify.py` - confirms output row counts after a run.

## 1. Install Termux

Install Termux from F-Droid or the official Termux GitHub release. Avoid the old Play Store Termux build because it is usually outdated.

Open Termux and run:

```bash
termux-setup-storage
```

Accept the storage permission prompt.

## 2. Copy Or Clone The Project

If you copied the folder into Android Downloads:

```bash
cd ~/storage/downloads/cs2_sticker_tracker
```

If you use Git:

```bash
pkg update -y
pkg install -y git
git clone <your-repo-url> ~/cs2_sticker_tracker
cd ~/cs2_sticker_tracker
```

Make sure your `data/scores.csv` is present because the analyzer uses your manual visual grades.

## 3. Install Requirements

```bash
bash android_setup.sh
```

This installs Python plus numpy/pandas from Termux packages. The setup script intentionally avoids global `pip` because Termux blocks it and global pip can break the package-managed Python. No BeautifulSoup package is required.

If you see `Installing pip packages is forbidden`, update this project and rerun:

```bash
bash android_setup.sh
```

Do not run `pip install pandas` globally in Termux.

## 4. Run The Full Pipeline

```bash
bash android_run.sh
```

This runs:

```bash
python android_collect.py
python analyze.py
python visualize.py
python android_verify.py
```

Expected output:

- `data/latest_snapshot.csv`
- `data/latest_history.csv`
- `data/history_points.csv`
- `analyze/latest_analysis_clean.csv`
- `visualized/sticker_dashboard.html`

On a phone, the first full run can take several minutes. It is normal.

## 5. Open The Dashboard

After the run finishes:

```bash
bash android_serve.sh
```

Then open this URL in Chrome/Firefox on the same phone:

```text
http://127.0.0.1:8765/visualized/sticker_dashboard.html
```

Keep Termux open while viewing the page. Press `Ctrl+C` in Termux to stop the server.

## Useful Commands

Lower metadata pressure if CS2Tokens rate-limits the phone:

```bash
python android_collect.py --metadata-workers 2
python analyze.py
python visualize.py
```

Debug only the first browse page:

```bash
python android_collect.py --max-pages 1 --no-metadata --no-history
```

Do not use the debug command for real decisions because it collects only part of the market.

Run the full pipeline with lower metadata concurrency:

```bash
bash android_run.sh --metadata-workers 2
```

Verify the latest output at any time:

```bash
python android_verify.py
```

## Expected Full Collection

For Cologne 2026, a full successful collection should show:

- `772` total stickers.
- `193` Paper.
- `193` Foil.
- `193` Holo.
- `193` Gold.
- `772` metadata rows with status `ok`.
- History for all `772` stickers.

## Notes

The Android collector is intentionally separate from `collect.py`. Desktop `collect.py` uses Playwright and is still useful on PC. Android `android_collect.py` is lighter and better suited to Termux.

The dashboard is static HTML, but opening it through `android_serve.sh` is more reliable than opening the file directly from Android storage.
