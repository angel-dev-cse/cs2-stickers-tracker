@echo off
cd /d "%~dp0"
python collect.py && python analyze.py && python visualize.py

REM Play a done sound
powershell -Command "[System.Media.SystemSounds]::Asterisk.Play()"

REM Open the dashboard in the default browser
start "" "E:\Codes\cs2_sticker_tracker\visualized\sticker_dashboard.html"