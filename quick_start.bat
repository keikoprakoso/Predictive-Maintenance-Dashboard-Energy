@echo off
REM 🚀 Quick Start Script for Predictive Maintenance Dashboard (Windows)
REM Author: Keiko Rafi Ananda Prakoso

echo 🏭 Predictive Maintenance Dashboard - Quick Start
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ Python and pip found

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
) else (
    echo ✅ Dependencies installed successfully
)

REM Run the complete pipeline
echo 🔄 Running complete pipeline...
python scripts/run_pipeline.py

if errorlevel 1 (
    echo ❌ Pipeline failed
    pause
    exit /b 1
) else (
    echo ✅ Pipeline completed successfully
)

REM Launch dashboard
echo 🌐 Launching dashboard...
echo 📊 Dashboard will be available at: http://localhost:8501
echo 🔄 Press Ctrl+C to stop the dashboard
echo.

streamlit run src/dashboard.py

pause 