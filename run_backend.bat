@echo off
REM Starts the Cognify-AI backend on Windows.
REM Double-click this file, or run it from a terminal in the project folder.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
echo Starting Cognify-AI on http://localhost:8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
