@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul

where pixi >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :pixi_found

echo pixi not found. Trying to install...

REM Try winget first (built into Windows 11, no script execution needed)
where winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Attempting install via winget...
    winget install prefix-dev.pixi --accept-package-agreements --accept-source-agreements
    if %ERRORLEVEL% EQU 0 goto :refresh_path
    echo winget install failed, trying direct download...
)

REM Fall back: download pixi.exe directly without running any script
echo Downloading pixi.exe directly from GitHub...
if not exist "%USERPROFILE%\.pixi\bin" mkdir "%USERPROFILE%\.pixi\bin"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Invoke-WebRequest -Uri 'https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe' -OutFile '%USERPROFILE%\.pixi\bin\pixi.exe'"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: All automatic install methods failed.
    echo Please install pixi manually:
    echo   1. Go to https://pixi.sh
    echo   2. Download and run the Windows installer
    echo   3. Re-run this bat file
    pause
    exit /b 1
)

:refresh_path
set "PATH=%USERPROFILE%\.pixi\bin;%PATH%"

where pixi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pixi installed but still not found on PATH. Restart your terminal and try again.
    pause
    exit /b 1
)

:pixi_found
cd /d "%~dp0"
echo Starting NucLogic via pixi...
pixi run start
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: pixi run start failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)
