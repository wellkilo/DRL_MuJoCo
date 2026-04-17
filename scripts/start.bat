@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
cd /d "%REPO_ROOT%"

set "ENV_NAME=drl-arm"

echo ======================================
echo DRL MuJoCo - Training Launcher
echo ======================================

:: Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'conda' command not found.
    echo Please install Miniforge/Miniconda first.
    echo Download from: https://github.com/conda-forge/miniforge
    exit /b 1
)

:: Initialize conda for batch scripting (works from any CMD, not just Anaconda Prompt)
echo Initializing conda...
for /f "delims=" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
if not exist "%CONDA_BASE%\condabin\conda_hook.bat" (
    echo ERROR: conda installation found at %CONDA_BASE% but conda_hook.bat is missing.
    exit /b 1
)
call "%CONDA_BASE%\condabin\conda_hook.bat"

:: Activate conda environment
echo Activating conda environment: %ENV_NAME%
call conda activate "%ENV_NAME%"
if errorlevel 1 (
    echo ERROR: Conda environment '%ENV_NAME%' not found.
    echo Please run 'scripts\build.bat' first to set up the environment.
    exit /b 1
)

:: Create output directory if it doesn't exist
if not exist "%REPO_ROOT%\output" mkdir "%REPO_ROOT%\output"

echo.
echo Select mode:
echo 1) Run distributed training (8 actors)
echo 2) Run single-agent training (1 actor)
echo 3) Plot training curves
echo 4) Plot comparison curves
echo 5) Launch Web UI (FastAPI backend)
echo 6) Launch Next.js Dev Server + Web UI (recommended)
set /p "choice=Enter choice [1]: "
if "!choice!"=="" set "choice=1"

if "!choice!"=="1" (
    echo.
    echo Starting distributed training...
    python "%REPO_ROOT%\main.py"
) else if "!choice!"=="2" (
    echo.
    echo Starting single-agent training...
    python "%REPO_ROOT%\main.py" "%REPO_ROOT%\config\config_single.yaml"
) else if "!choice!"=="3" (
    echo.
    echo Plotting training curves...
    python "%REPO_ROOT%\scripts\plot_training.py" --all
    echo Curves saved to: output\figures\
) else if "!choice!"=="4" (
    echo.
    echo Plotting comparison curves...
    python "%REPO_ROOT%\scripts\plot_comparison.py"
    echo Curves saved to: output\comparison_curves.png
) else if "!choice!"=="5" (
    echo.
    echo Launching Web UI...
    echo Open your browser and go to: http://127.0.0.1:8000
    python "%REPO_ROOT%\web\server.py"
) else if "!choice!"=="6" (
    echo.
    where npm >nul 2>&1
    if errorlevel 1 (
        echo ERROR: npm not found.
        echo Install Node.js from https://nodejs.org/
        exit /b 1
    )
    cd /d "%REPO_ROOT%\web"
    if not exist "node_modules" (
        echo Installing npm dependencies...
        call npm install
    )
    echo Launching Next.js Dev Server + Web UI...
    echo Open your browser and go to: http://localhost:3000
    echo Next.js will proxy /api and /ws to FastAPI backend at http://127.0.0.1:8000
    echo Starting FastAPI backend in background...
    cd /d "%REPO_ROOT%"
    start "DRL_MuJoCo_FastAPI" /min "%CONDA_PREFIX%\python.exe" "%REPO_ROOT%\web\server.py"
    echo Waiting for FastAPI backend to start...
    set "backend_ready=0"
    for /l %%i in (1,1,15) do (
        if !backend_ready! equ 0 (
            powershell -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
            if !errorlevel! equ 0 (
                echo FastAPI backend is ready!
                set "backend_ready=1"
            ) else (
                echo   Waiting... (%%i/15)
                timeout /t 2 /nobreak >nul
            )
        )
    )
    if !backend_ready! equ 0 (
        echo WARNING: FastAPI backend may not have started in time.
        echo Check the DRL_MuJoCo_FastAPI window for errors.
    )
    cd /d "%REPO_ROOT%\web"
    echo Starting Next.js dev server...
    call npm run dev
    echo Stopping FastAPI backend...
    taskkill /fi "WINDOWTITLE eq DRL_MuJoCo_FastAPI" >nul 2>&1
    cd /d "%REPO_ROOT%"
) else (
    echo ERROR: Invalid choice.
    exit /b 1
)

echo.
echo Done!
echo.

endlocal