@echo off
REM Set paths relative to the batch file location
set SCRIPT_DIR=%~dp0
set ENV_YML=%SCRIPT_DIR%miscellaneous\environment.yml
set SAM2=C:\repos\sam2

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
set ENV_PATH=%CONDA%\envs\NucLogicEnv

REM Step 1: Activate Conda base
echo Activating conda base from %CONDA%
call "%CONDA%\Scripts\activate.bat" "%CONDA%"

REM Step 2: Check if environment exists
IF NOT EXIST "%ENV_PATH%" (
    echo Environment not found. Creating it now... This might take a few minutes
    call conda env create --prefix "%ENV_PATH%" --file "%ENV_YML%"
    call conda activate "%ENV_PATH%"
    echo Installing PyTorch with CUDA 12.8 support...
    call pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
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

REM Step 3: Clone sam2 if not present, then always ensure it is installed correctly
IF NOT EXIST "%SAM2%" (
    echo Cloning sam2...
    git clone https://github.com/facebookresearch/sam2.git "%SAM2%"
)
echo Installing/verifying sam2 and its dependencies...
cd /d "%SAM2%"
call pip install -e .
cd /d "%SCRIPT_DIR%"

REM Step 4: Change to project folder
cd /d "%SCRIPT_DIR%"

REM Suppress OpenMP conflict between numpy and torch bundled runtimes
set KMP_DUPLICATE_LIB_OK=TRUE

REM Step 5: Run the app
echo Starting NucLogic
call python "PySide2\app.py"

REM Step 6: Keep window open
cmd /k