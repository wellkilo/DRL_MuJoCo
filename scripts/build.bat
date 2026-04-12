@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
cd /d "%REPO_ROOT%"

set "ENV_NAME=drl-arm"
set "PYTHON_VERSION=3.9"

echo ======================================
echo DRL MuJoCo - Environment Setup
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

echo Creating/activating conda environment: %ENV_NAME%
conda env list | findstr /b "%ENV_NAME% " >nul 2>&1
if errorlevel 1 (
    echo Creating new conda environment with Python %PYTHON_VERSION%...
    call conda create -y -n "%ENV_NAME%" python="%PYTHON_VERSION%"
) else (
    echo Conda environment '%ENV_NAME%' already exists.
)

call conda activate "%ENV_NAME%"
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    exit /b 1
)

echo Installing Python packages...
pip install --upgrade pip
pip install -r "%REPO_ROOT%\requirements.txt"

echo.
echo ======================================
echo Additional Components (Optional)
echo ======================================
echo.

:: Ask about Next.js frontend
set /p "build_next=Build Next.js frontend? [y/N]: "
if /i "!build_next!"=="y" (
    where npm >nul 2>&1
    if errorlevel 1 (
        echo WARNING: npm not found. Skipping Next.js frontend build.
        echo Install Node.js from https://nodejs.org/
    ) else (
        echo Building Next.js frontend...
        cd /d "%REPO_ROOT%\web"
        call npm install
        call npm run build
        cd /d "%REPO_ROOT%"
        echo Next.js frontend built successfully!
    )
)

:: Ask about Rust Buffer
set /p "build_rust=Build Rust Buffer? [y/N]: "
if /i "!build_rust!"=="y" (
    where cargo >nul 2>&1
    if errorlevel 1 (
        echo WARNING: cargo not found. Skipping Rust Buffer build.
        echo Install Rust from https://www.rust-lang.org/tools/install
    ) else (
        echo Building Rust Buffer...
        cd /d "%REPO_ROOT%\rust_buffer"
        pip install maturin
        maturin develop --release
        cd /d "%REPO_ROOT%"
        echo Rust Buffer built successfully!
    )
)

echo.
echo ======================================
echo Setup complete!
echo ======================================
echo.
echo To use this environment:
echo   conda activate %ENV_NAME%
echo.
echo To run training/UI:
echo   scripts\start.bat
echo.
echo To build Next.js frontend manually:
echo   cd web ^& npm install ^& npm run build
echo.
echo To build Rust Buffer manually:
echo   cd rust_buffer ^& pip install maturin ^& maturin develop --release
echo.

endlocal