@echo off
:: Check if the script is run as Administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrator privileges.
    echo Please run this script as an administrator.
    pause
    exit /b
)

cd /d %~dp0
type about.nfo
@echo off
call conda activate memo >nul
python gradio_app.py
