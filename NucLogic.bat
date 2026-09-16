@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul

REM NucLogic launcher.
REM
REM Building the environment needs no administrator rights of its own: pixi
REM writes only inside this folder. Admin is used for one thing only, granting
REM every Windows account on this PC access to the installation. So elevation is
REM offered, not required: decline it and setup carries on for this user alone.
REM
REM The elevated copy is started as "NucLogic.bat --setup". It builds the
REM environment, grants access and exits without launching anything, so NucLogic
REM always runs as the person who double-clicked it. That matters: an elevated
REM process cannot see the launching user's mapped network drives, so a data
REM folder on Z:\ would be invisible in the folder browser.

set "SETUP_ONLY="
if /I "%~1"=="--setup" set "SETUP_ONLY=1"

REM Use one shared package cache inside the install folder instead of each user's
REM (or the elevated admin's) %LOCALAPPDATA%\rattler. This keeps the cache on the
REM same drive as the environment (required for rattler's hardlink/rename step),
REM makes it shared across all accounts, and gets covered by the icacls grant
REM below.
set "PIXI_CACHE_DIR=%~dp0pixi_cache"
cd /d "%~dp0"
set "ENV_DIR=%~dp0.pixi\envs\default"

set "ELEVATED="
net session >nul 2>&1
if !ERRORLEVEL! EQU 0 set "ELEVATED=1"

if exist "%ENV_DIR%" (
    if defined SETUP_ONLY exit /b 0
    goto :launch
)

REM ------------------------------------------------------------- first run ---
REM Offer to run setup elevated so all accounts get access. Declining is fine.
if not defined SETUP_ONLY if not defined ELEVATED (
    echo First-time setup detected.
    echo.
    echo NucLogic can be installed for every user on this PC, which needs
    echo administrator rights. A User Account Control prompt will appear.
    echo If you decline it, setup continues for your account only.
    echo.
    powershell -NoProfile -Command ^
        "try { $p = Start-Process -FilePath '%~f0' -ArgumentList '--setup' -Verb RunAs -Wait -PassThru; exit $p.ExitCode } catch { exit 1223 }"
    set "ELEV_RESULT=!ERRORLEVEL!"

    if exist "%ENV_DIR%" goto :launch

    echo.
    if "!ELEV_RESULT!"=="1223" (
        echo No administrator rights were given, so NucLogic will be installed
        echo for your account only. It will work normally.
        echo To share it with other Windows accounts later, right-click
        echo NucLogic.bat and choose "Run as administrator".
    ) else (
        echo Setup with administrator rights did not finish ^(code !ELEV_RESULT!^).
        echo Continuing for your account only.
    )
    echo.
)

call :ensure_pixi
if !ERRORLEVEL! NEQ 0 goto :die

echo Building the environment, this may take several minutes...
call "!PIXI!" install
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ERROR: pixi install failed with exit code !ERRORLEVEL!.
    echo If the message above mentions git, install Git for Windows from
    echo https://git-scm.com/download/win and run NucLogic.bat again.
    goto :die
)

REM Grant every user on this PC access. The (OI)(CI) flags make the grant
REM inheritable, which covers plain files created later. It does NOT cover
REM packages that pixi/rattler hardlinks in from pixi_cache: a hardlink keeps the
REM cache file's own DACL instead of inheriting this folder's. So the in-app
REM updater re-applies this same grant itself after its pixi install (see
REM utils/updater.py _grant_new_files_access); that is what keeps updates working
REM for every account.
if defined ELEVATED (
    echo Granting all users full access to the installation...
    REM Take ownership first so the DACL can be rewritten even on files created
    REM by another account or with restrictive permissions.
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
) else (
    echo Installed for this user account only.
)

if defined SETUP_ONLY (
    echo Setup complete.
    exit /b 0
)

REM ---------------------------------------------------------------- launch ---
:launch
call :ensure_pixi
if !ERRORLEVEL! NEQ 0 goto :die

echo Starting NucLogic via pixi...
call "!PIXI!" run start
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ERROR: pixi run start failed with exit code !ERRORLEVEL!.
    goto :die
)
exit /b 0

:die
echo.
pause
exit /b 1

REM --------------------------------------------------------------------------
REM Locate pixi, downloading it if needed, and confirm the binary actually runs.
:ensure_pixi
set "PIXI="
if exist "%~dp0tools\pixi.exe" (
    set "PIXI=%~dp0tools\pixi.exe"
    goto :ensure_pixi_verify
)

where pixi >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    set "PIXI=pixi"
    goto :ensure_pixi_verify
)

echo pixi not found. Downloading it into the tools folder...
if not exist "%~dp0tools" mkdir "%~dp0tools" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { Invoke-WebRequest -Uri 'https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe' -OutFile '%~dp0tools\pixi.exe' } catch { exit 1 }"
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ERROR: could not download pixi. Common causes:
    echo   - no internet connection, or a proxy blocking github.com
    echo   - this folder is not writable by your account ^(for example under
    echo     Program Files^) -- re-run NucLogic.bat as administrator
    echo Alternatively install pixi yourself from https://pixi.sh and run this
    echo file again.
    exit /b 1
)
set "PIXI=%~dp0tools\pixi.exe"

:ensure_pixi_verify
REM A captive portal or proxy can answer with an HTML page and HTTP 200, leaving
REM a "pixi.exe" that is not an executable. Catch that here rather than letting
REM it fail confusingly inside pixi install.
call "!PIXI!" --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ERROR: the pixi executable is present but does not run.
    if exist "%~dp0tools\pixi.exe" (
        del /q "%~dp0tools\pixi.exe" >nul 2>&1
        echo The downloaded copy has been deleted; run NucLogic.bat again to
        echo download it afresh.
    )
    exit /b 1
)
exit /b 0
