#!/bin/bash

# 🚀 Quick Start Script for Predictive Maintenance Dashboard
# Author: Keiko Rafi Ananda Prakoso

echo "🏭 Predictive Maintenance Dashboard - Quick Start"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Run the complete pipeline
echo "🔄 Running complete pipeline..."
python3 scripts/run_pipeline.py

if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully"
else
    echo "❌ Pipeline failed"
    exit 1
fi

# Launch dashboard
echo "🌐 Launching dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🔄 Press Ctrl+C to stop the dashboard"
echo ""

streamlit run src/dashboard.py 