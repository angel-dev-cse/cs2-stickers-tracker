@echo off
cd /d "%~dp0"
python collect.py && python analyze.py && python visualize.py

REM Play a done sound
powershell -Command "[System.Media.SystemSounds]::Asterisk.Play()"

REM Open the dashboard in the default browser
start "CS2 Sticker Dashboard Server" /min python inventory_server.py 8765
timeout /t 2 >nul
start "" "http://127.0.0.1:8765/visualized/sticker_dashboard.html"
