@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul

REM Look for pixi bundled inside the project first (works for any user account)
set "PIXI_LOCAL=%~dp0tools\pixi.exe"
if exist "%PIXI_LOCAL%" (
    set "PIXI=%PIXI_LOCAL%"
    goto :pixi_found
)

REM Fall back to pixi already on the system PATH
where pixi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PIXI=pixi"
    goto :pixi_found
)

echo pixi not found. Downloading into project tools folder...
if not exist "%~dp0tools" mkdir "%~dp0tools"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Invoke-WebRequest -Uri 'https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe' -OutFile '%~dp0tools\pixi.exe'"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Download failed. Install pixi manually from https://pixi.sh
    pause
    exit /b 1
)
set "PIXI=%~dp0tools\pixi.exe"

:pixi_found
cd /d "%~dp0"
echo Starting NucLogic via pixi...
"%PIXI%" run start
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: pixi run start failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)
