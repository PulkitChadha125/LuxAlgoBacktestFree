@echo off
setlocal

cd /d "%~dp0"

echo ============================================
echo Fyers Streamlit Backtest - Auto Runner
echo ============================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] .venv not found. Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] .venv already exists.
)

echo [2/4] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [3/4] Installing/updating requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo [4/4] Starting Streamlit app...
streamlit run app.py

endlocal
