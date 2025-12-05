@echo off
REM Set paths relative to the batch file location
set SCRIPT_DIR=%~dp0
set CONDA=C:\ProgramData\miniconda3
set ENV_PATH=%CONDA%\envs\organoid_segmenter
set ENV_YML=%SCRIPT_DIR%miscellaneous\environment.yml
set REQUIREMENTS=%SCRIPT_DIR%miscellaneous\requirements.txt
set SAM2=C:\repos\sam2

REM Step 1: Activate Conda base
echo Activating conda
echo "%CONDA%\Scripts\activate.bat" "%CONDA%"
call "%CONDA%\Scripts\activate.bat" "%CONDA%"

REM Step 2: Check if environment exists
IF NOT EXIST "%ENV_PATH%" (
    echo Environment not found. Creating it now... This might take a few minutes
    call conda env create --prefix "%ENV_PATH%" --file "%ENV_YML%"
    call conda activate "%ENV_PATH%"
    echo activated environment
    call pip install -r "%REQUIREMENTS%" 
    echo Finished pip install
) ELSE (
    echo Environment already exists. Starting now...
    echo Starting environment "%ENV_PATH%"
    call conda activate "%ENV_PATH%"
)

REM Ensure parent folder for SAM2 exists
if not exist "C:\repos\" (
    mkdir "C:\repos"
)

REM Step 3: Clone and install sam2 in env if not already installed
echo going to check if sam2 is installed
IF NOT EXIST "%SAM2%" (
    echo Cloning and installing sam2 into env...
    git clone https://github.com/facebookresearch/sam2.git "%SAM2%"
    cd /d "%SAM2%"
    call pip install -e .
    cd /d "%SCRIPT_DIR%"
) ELSE (
    echo sam2 already exists in env. Skipping installation.
)

REM Step 4: Change to project folder
cd /d "%SCRIPT_DIR%"

REM Step 5: Run the script
echo running shiny
call python -m shiny run --reload --launch-browser "shiny\app.py"



REM Step 6: Keep window open
cmd /k