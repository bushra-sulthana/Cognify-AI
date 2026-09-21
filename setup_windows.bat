@echo off
REM One-time setup for Windows: virtual environment + dependencies.
cd /d "%~dp0"
echo Creating virtual environment ...
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing dependencies (this downloads PyTorch and may take several minutes) ...
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Setup finished. Next steps:
echo   1. ollama pull llama3.2:3b
echo   2. run_backend.bat
echo   3. open http://localhost:8000
pause
