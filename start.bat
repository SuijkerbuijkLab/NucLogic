@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Set paths relative to the batch file location
set "SCRIPT_DIR=%~dp0"
set "ENV_YML=%SCRIPT_DIR%miscellaneous\environment.yml"
set "SAM2=C:\repos\sam2"

REM Configure where log files are stored
set "LOG_DIR=%SCRIPT_DIR%logs"

REM Wrapper mode: run the script once and capture all output to a timestamped log file
if defined LOG_WRAPPED goto :log_ready

if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
)

set "TIMESTAMP="
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd_HH-mm-ss')" 2^>nul') do set "TIMESTAMP=%%I"
if not defined TIMESTAMP (
    set "TIMESTAMP=!DATE!_!TIME!"
    set "TIMESTAMP=!TIMESTAMP: =0!"
    set "TIMESTAMP=!TIMESTAMP::=-!"
    set "TIMESTAMP=!TIMESTAMP:/=-!"
    set "TIMESTAMP=!TIMESTAMP:.=-!"
    set "TIMESTAMP=!TIMESTAMP:,=-!"
)
set "LOG_FILE=%LOG_DIR%\log_!TIMESTAMP!.txt"

set "LOG_WRAPPED=1"
echo Logging to "%LOG_FILE%"

REM Download wtee.exe if not available
if not exist "%SCRIPT_DIR%wtee.exe" if not exist "%SCRIPT_DIR%tee.exe" (
    echo Downloading wtee.exe...
    powershell -NoProfile -Command "& { Invoke-WebRequest -Uri 'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/wintee/wtee.exe' -OutFile '%SCRIPT_DIR%wtee.exe' }"
)

set "WTEE="
if exist "%SCRIPT_DIR%wtee.exe" set "WTEE=%SCRIPT_DIR%wtee.exe"
if not defined WTEE if exist "%SCRIPT_DIR%tee.exe" set "WTEE=%SCRIPT_DIR%tee.exe"
if not defined WTEE (
    for /f "delims=" %%W in ('where wtee.exe 2^>nul') do (
        set "WTEE=%%W"
        goto :wtee_found
    )
)
:wtee_found
if not defined WTEE (
    for /f "delims=" %%W in ('where tee.exe 2^>nul') do (
        set "WTEE=%%W"
        goto :wtee_found
    )
)

if defined WTEE (
    cmd /d /c "set LOG_WRAPPED=1& call "%~f0" %*" 2>&1 | "%WTEE%" "%LOG_FILE%"
) else (
    echo NOTE: wtee.exe not found, logging to file only.
    cmd /d /c "set LOG_WRAPPED=1& call "%~f0" %*" > "%LOG_FILE%" 2>&1
)
exit /b %ERRORLEVEL%

:log_ready

REM Force UTF-8 output so progress bars and unicode don't crash
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM Detect conda installation path dynamically
set CONDA=

REM 1. Use CONDA_EXE if set (most reliable — set by conda at install time)
if defined CONDA_EXE (
    for %%i in ("%CONDA_EXE%\..\..\..") do set CONDA=%%~fi
    goto :conda_found
)

REM 2. Fall back to known default install locations
for %%P in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\Miniconda3"
    "C:\ProgramData\anaconda3"
    "C:\ProgramData\Anaconda3"
) do (
    if exist "%%~P\Scripts\activate.bat" (
        set CONDA=%%~P
        goto :conda_found
    )
)

echo ERROR: conda not found. Please install Miniconda or Anaconda.
pause
exit /b 1
:conda_found
set ENV_PATH=C:\Users\6331823\AppData\Local\anaconda3\envs\nuclei_segmenter

REM Step 1: Activate Conda base
echo Activating conda base from %CONDA%
call "%CONDA%\Scripts\activate.bat" "%CONDA%"

REM Step 2: Check if environment exists
IF NOT EXIST "%ENV_PATH%" (
    echo Environment not found. Creating it now... This might take a few minutes
    call conda env create --prefix "%ENV_PATH%" --file "%ENV_YML%"
    call conda activate "%ENV_PATH%"
    echo Installing PyTorch with CUDA 12.8 support...
    call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo Installing cellpose...
    call pip install cellpose
    echo Environment setup complete.
) ELSE (
    echo Environment already exists. Starting now...
    echo Starting environment "%ENV_PATH%"
    call conda activate "%ENV_PATH%"
)

REM Ensure parent folder for SAM2 exists
if not exist "C:\repos\" (
    mkdir "C:\repos"
)

REM Step 3: Clone sam2 if not present, then install if sam2 or its dependencies are missing
IF NOT EXIST "%SAM2%" (
    echo Cloning sam2...
    git clone https://github.com/facebookresearch/sam2.git "%SAM2%"
)
python -m pip show SAM-2 >nul 2>&1
set SAM2_INSTALLED=%ERRORLEVEL%
python -m pip show hydra-core >nul 2>&1
set HYDRA_INSTALLED=%ERRORLEVEL%
set NEED_INSTALL=0
IF %SAM2_INSTALLED% NEQ 0 set NEED_INSTALL=1
IF %HYDRA_INSTALLED% NEQ 0 set NEED_INSTALL=1
IF %NEED_INSTALL%==1 (
    echo sam2 or dependencies not fully installed. Installing...
    cd /d "%SAM2%"
    call pip install -e .
    cd /d "%SCRIPT_DIR%"
) ELSE (
    echo sam2 already installed. Skipping.
)

REM Step 4: Change to project folder
cd /d "%SCRIPT_DIR%"

REM Suppress OpenMP conflict between numpy and torch bundled runtimes
set KMP_DUPLICATE_LIB_OK=TRUE

REM Step 5: Run the app
echo Starting NucLogic
call python "PySide2\app.py"