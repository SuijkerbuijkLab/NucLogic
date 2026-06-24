@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul

REM Use one shared package cache inside the install folder instead of each user's
REM (or the elevated admin's) %LOCALAPPDATA%\rattler. This keeps the cache on the
REM same drive as the environment (required for rattler's hardlink/rename step),
REM makes it shared across all accounts, and gets covered by the icacls grant below.
set "PIXI_CACHE_DIR=%~dp0pixi_cache"

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

REM First-time setup
REM When the environment doesn't exist yet, build it and grant every user on this
REM PC full access to the install folder. The permissions are INHERITABLE (the
REM (OI)(CI) flags), so files created later by the in-app updater, even when it
REM wipes and rebuilds .pixi\envs automatically inherit them. That means this
REM step does NOT need to run again after an update.
if not exist "%~dp0.pixi\envs\default" (
    REM First-time setup needs admin rights for takeown/icacls below. If we're not
    REM elevated, relaunch this script as administrator and let that copy do the
    REM setup. Normal launches (env already built) never reach here, so regular
    REM users can start the app without admin.
    net session >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo First-time setup needs administrator rights to grant all users access.
        echo Requesting elevation ^(a User Account Control prompt will appear^)...
        powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
        exit /b
    )

    echo First run detected. Building the environment, this may take several minutes...
    "%PIXI%" install
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo ERROR: pixi install failed with exit code !ERRORLEVEL!.
        pause
        exit /b !ERRORLEVEL!
    )

    echo Granting all users full access to the installation...
    REM Take ownership first so the DACL can be rewritten even on files created
    REM by another account or with restrictive permissions. Requires admin rights;
    REM if it fails, the icacls warning below will tell the user to elevate.
    takeown /F "%~dp0." /R /D Y >nul 2>&1

    REM *S-1-5-32-545 is the BUILTIN\Users SID ^(locale-independent^).
    REM /T recurse, /C continue past locked files, /Q hide per-file success spam
    REM ^(errors are still shown^).
    icacls "%~dp0." /grant "*S-1-5-32-545:(OI)(CI)F" /T /C /Q
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo WARNING: some files could not be granted access ^(often just a file
        echo briefly locked by antivirus/indexing^). If other users hit "access
        echo denied" errors when launching, re-run NucLogic.bat as administrator
        echo ^(right-click ^> Run as administrator^).
    )
)

echo Starting NucLogic via pixi...
"%PIXI%" run start
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: pixi run start failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)
