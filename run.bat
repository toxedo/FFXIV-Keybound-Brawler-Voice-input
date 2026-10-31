@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

rem ---- FF14 runs elevated, so this must too ----
powershell -NoProfile -Command "if (([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { exit 0 } else { exit 1 }"
if %errorlevel% equ 0 goto run

if "%VV_TRY%"=="1" goto failed

if exist "%~dp0.elevating.tmp" (
    del "%~dp0.elevating.tmp" >nul 2>&1
    echo [WARN] Previous elevation did not complete. Running WITHOUT admin.
    echo        Keystrokes will not reach the game.
    goto run
)

echo.
echo   FF14 runs as administrator, so this tool must too.
echo   Requesting administrator rights... (click Yes on the UAC prompt)
echo.
echo 1> "%~dp0.elevating.tmp"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','set \"VV_TRY=1\"&& call \"%~f0\" %*' -Verb RunAs"
exit /b

:failed
del "%~dp0.elevating.tmp" >nul 2>&1
echo.
echo [ERROR] Elevation failed. Stopped to avoid endless UAC popups.
echo         Right-click run.bat and choose "Run as administrator".
echo.
pause
exit /b 1

:run
del "%~dp0.elevating.tmp" >nul 2>&1
set PYTHONIOENCODING=utf-8
python voice_input.py %*
echo.
echo [stopped] press any key to close
pause >nul
