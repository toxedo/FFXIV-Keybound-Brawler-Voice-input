@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================
echo  FF14 Voice Input - Setup
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] "python" not found in PATH.
    echo.
    echo   Install Python 3.10 or newer from:
    echo   https://www.python.org/downloads/
    echo   IMPORTANT: check "Add python.exe to PATH" during install.
    echo.
    pause
    exit /b 1
)

python --version
echo.
echo Installing dependencies (sounddevice / faster-whisper / pypinyin)...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed. Likely a network issue.
    echo   Try a proxy or a domestic mirror, then run this again.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Done.
echo ============================================
echo.
echo  Next step: double-click run.bat
echo.
echo  First launch downloads the model (about 460 MB for "small").
echo  If the download stalls, close the window and run:
echo      setx HF_ENDPOINT https://hf-mirror.com
echo  then reopen run.bat.
echo.
pause
