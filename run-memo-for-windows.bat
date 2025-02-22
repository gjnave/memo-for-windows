cd /d %~dp0
type about.nfo
@echo off
call conda activate memo >nul
python gradio_app.py
