# 🛠️ Installation Guide - Predictive Maintenance Dashboard

## 📋 Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **pip** package manager
- **Git** (for cloning the repository)

## 🚀 Quick Installation

### Option 1: Basic Installation (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd "Predictive Maintenance Dashboard (Energy Sector)"

# Install core dependencies (without Prophet)
pip install -r requirements.txt
```

### Option 2: Full Installation (with Prophet)
```bash
# Install Prophet separately (may require additional setup)
pip install prophet
pip install -r requirements.txt
```

## ⚠️ Prophet Installation Issues

Prophet can be challenging to install due to its dependencies. Here are solutions:

### Solution 1: Use Conda (Recommended)
```bash
# Install conda if you don't have it
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create a new environment
conda create -n predictive-maintenance python=3.9
conda activate predictive-maintenance

# Install Prophet via conda
conda install -c conda-forge prophet

# Install other dependencies
pip install -r requirements.txt
```

### Solution 2: Use pip with Build Tools
```bash
# On Windows, install Visual Studio Build Tools
# On macOS, install Xcode Command Line Tools
xcode-select --install

# On Linux, install build essentials
sudo apt-get install build-essential

# Install Prophet
pip install prophet
```

### Solution 3: Alternative Time Series Libraries
If Prophet installation fails, the project will automatically use alternative methods:

```bash
# Install alternative time series libraries
pip install statsmodels
pip install pmdarima
pip install arch

# The project will use Exponential Smoothing and ARIMA models instead
```

## 📦 Package-by-Package Installation

If you encounter issues with specific packages, install them individually:

```bash
# Core data science
pip install pandas==2.1.4
pip install numpy==1.24.3
pip install scikit-learn==1.3.2

# Time series (choose one)
pip install statsmodels==0.14.0  # Alternative to Prophet
# OR
pip install prophet==1.1.4       # If installation succeeds

# Dashboard
pip install streamlit==1.28.1
pip install plotly==5.17.0
pip install altair==5.1.2

# Utilities
pip install pyyaml==6.0.1
pip install python-dateutil==2.8.2
pip install tqdm==4.66.1
pip install joblib==1.3.2

# Development
pip install jupyter==1.0.0
pip install ipykernel==6.25.2
```

## 🔧 Platform-Specific Instructions

### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install via conda (recommended)
conda install -c conda-forge prophet

# OR install via pip
pip install prophet
```

### Windows
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install via conda (recommended)
conda install -c conda-forge prophet

# OR install via pip
pip install prophet
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Install Prophet
pip install prophet
```

## 🐍 Virtual Environment Setup

### Using venv (Python 3.3+)
```bash
# Create virtual environment
python3 -m venv predictive-maintenance-env

# Activate environment
# On macOS/Linux:
source predictive-maintenance-env/bin/activate
# On Windows:
predictive-maintenance-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using conda
```bash
# Create conda environment
conda create -n predictive-maintenance python=3.9

# Activate environment
conda activate predictive-maintenance

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running the Project

### Without Prophet (Basic Setup)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data generation
python3 src/data_generation.py

# 3. Run data processing
python3 src/data_processing.py

# 4. Run ML models (will use alternative methods)
python3 src/ml_models.py

# 5. Launch dashboard
streamlit run src/dashboard.py
```

### With Prophet (Full Setup)
```bash
# 1. Install Prophet first
conda install -c conda-forge prophet
# OR
pip install prophet

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Run complete pipeline
python3 run_pipeline.py
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Prophet Installation Fails
**Error:** `Failed to build wheel for prophet`

**Solutions:**
- Use conda: `conda install -c conda-forge prophet`
- Install build tools for your platform
- Use alternative time series libraries (project supports this)

#### 2. Import Errors
**Error:** `ModuleNotFoundError: No module named 'prophet'`

**Solution:** The project will automatically use alternative methods. No action needed.

#### 3. Streamlit Issues
**Error:** `streamlit: command not found`

**Solution:**
```bash
pip install streamlit
# OR
python3 -m streamlit run src/dashboard.py
```

#### 4. Jupyter Issues
**Error:** `jupyter: command not found`

**Solution:**
```bash
pip install jupyter
# OR
python3 -m jupyter notebook
```

### Performance Issues

#### 1. Slow Data Generation
- Reduce the number of turbines in `config.yaml`
- Reduce the date range for testing

#### 2. Memory Issues
- Process data in smaller chunks
- Use fewer features in the ML models

#### 3. Dashboard Loading Slowly
- Reduce the number of data points displayed
- Use data sampling for large datasets

## 📊 Verification

After installation, verify everything works:

```bash
# Test data generation
python3 src/data_generation.py

# Check if files were created
ls -la data/raw/

# Test data processing
python3 src/data_processing.py

# Check processed data
ls -la data/processed/

# Test dashboard
streamlit run src/dashboard.py
```

## 🆘 Getting Help

If you encounter issues:

1. **Check the error messages** - They often contain helpful information
2. **Try the alternative installation methods** - Use conda instead of pip
3. **Use the basic setup** - Skip Prophet and use alternative time series methods
4. **Check platform-specific instructions** - Different OS may require different steps

## 📝 Notes

- **Prophet is optional** - The project works without it using alternative methods
- **Performance may vary** - Alternative methods may be slightly less accurate but still functional
- **All core features work** - Dashboard, risk assessment, and cost analysis work regardless of Prophet availability

---

*The project is designed to be robust and work with or without Prophet. If you can't install Prophet, the alternative time series methods will provide similar functionality.* 