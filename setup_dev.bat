@echo off
REM Setup script for local development on Windows
REM Run: setup_dev.bat

echo.
echo ======================================
echo 🚀 Setting up Land-Cover Segmentation API...
echo ======================================
echo.

REM Check Python version
echo ✓ Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment
echo ✓ Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo.  Virtual environment created
) else (
    echo.  Virtual environment already exists
)

REM Activate virtual environment
echo ✓ Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ✓ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo ✓ Installing dependencies...
pip install -r requirements.txt

REM Create directories
echo ✓ Creating necessary directories...
if not exist "workdir" mkdir workdir
if not exist "logs" mkdir logs
if not exist "output" mkdir output
if not exist "output\predicted_masks" mkdir output\predicted_masks
if not exist "output\prediction_plots" mkdir output\prediction_plots

REM Setup environment file
echo ✓ Setting up environment variables...
if not exist ".env" (
    copy .env.example .env
    echo.  Created .env file (please update GEMINI_API_KEY)
) else (
    echo.  .env file already exists
)

REM Run tests
echo ✓ Running tests...
pytest tests/ -v --tb=short
if errorlevel 1 (
    echo WARNING: Some tests failed, but this is okay for initial setup
)

REM Summary
echo.
echo ======================================
echo ✅ Setup completed successfully!
echo ======================================
echo.
echo Next steps:
echo 1. Update .env with your GEMINI_API_KEY
echo 2. Start the server:
echo    python -m uvicorn app.main:app --reload
echo 3. Open http://localhost:8000/docs in your browser
echo.
echo To deactivate virtual environment: deactivate
echo.

pause
