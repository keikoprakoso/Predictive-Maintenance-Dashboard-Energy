# 📁 Project Structure

## 🏗️ **Organized Directory Layout**

```
Predictive Maintenance Dashboard (Energy Sector)/
├── 📁 src/                          # Source code
│   ├── 📄 data_generation.py        # Mock data generation
│   ├── 📄 data_processing.py        # Data cleaning & feature engineering
│   ├── 📄 ml_models.py              # ML models & predictions
│   └── 📄 dashboard.py              # Streamlit dashboard
│
├── 📁 data/                         # Data storage
│   ├── 📁 raw/                      # Raw generated data
│   ├── 📁 processed/                # Cleaned & processed data
│   └── 📁 models/                   # Trained ML models
│
├── 📁 docs/                         # Documentation
│   ├── 📁 technical/                # Technical documentation
│   │   ├── 📄 PROJECT_DOCUMENTATION.md
│   │   ├── 📄 INSTALLATION_GUIDE.md
│   │   ├── 📄 DASHBOARD_FIXES.md
│   │   └── 📄 DASHBOARD_STATUS.md
│   ├── 📁 business/                 # Business documentation
│   │   ├── 📄 PROJECT_SUMMARY.md
│   │   ├── 📄 FINAL_SUMMARY.md
│   │   └── 📄 LINKEDIN_POST.md
│   └── 📁 user_guides/              # User guides (future)
│
├── 📁 scripts/                      # Utility scripts
│   └── 📄 run_pipeline.py           # Main pipeline runner
│
├── 📁 tests/                        # Testing
│   ├── 📄 test_dashboard.py         # Dashboard testing
│   ├── 📁 unit/                     # Unit tests (future)
│   └── 📁 integration/              # Integration tests (future)
│
├── 📁 notebooks/                    # Jupyter notebooks
│   └── 📄 exploratory_analysis.ipynb
│
├── 📁 assets/                       # Static assets
│   ├── 📁 images/                   # Images & charts
│   └── 📁 logos/                    # Company logos
│
├── 📁 deployment/                   # Deployment configs (future)
│
├── 📄 README.md                     # Main project README
├── 📄 requirements.txt              # Python dependencies
├── 📄 config.yaml                   # Configuration file
├── 📄 .gitignore                    # Git ignore rules
└── 📄 PROJECT_STRUCTURE.md          # This file
```

---

## 📋 **File Descriptions**

### **🔧 Source Code (`src/`)**
- **`data_generation.py`**: Generates mock turbine sensor data with realistic failure patterns
- **`data_processing.py`**: Cleans data, handles missing values, and engineers features
- **`ml_models.py`**: Implements ML models for time series forecasting and failure prediction
- **`dashboard.py`**: Streamlit web application for interactive dashboard

### **📊 Data (`data/`)**
- **`raw/`**: Contains generated mock sensor data
- **`processed/`**: Contains cleaned and feature-engineered data
- **`models/`**: Stores trained ML models and configurations

### **📚 Documentation (`docs/`)**
- **`technical/`**: Technical documentation, installation guides, and troubleshooting
- **`business/`**: Business summaries, ROI analysis, and portfolio materials
- **`user_guides/`**: User manuals and guides (for future use)

### **🛠️ Scripts (`scripts/`)**
- **`run_pipeline.py`**: Orchestrates the entire data pipeline from generation to dashboard

### **🧪 Testing (`tests/`)**
- **`test_dashboard.py`**: Tests dashboard functionality and data integrity
- **`unit/`**: Unit tests for individual components (future)
- **`integration/`**: Integration tests for the full pipeline (future)

### **📓 Notebooks (`notebooks/`)**
- **`exploratory_analysis.ipynb`**: Jupyter notebook for data exploration and analysis

### **🎨 Assets (`assets/`)**
- **`images/`**: Screenshots, charts, and visual assets
- **`logos/`**: Company and project logos

### **🚀 Deployment (`deployment/`)**
- Future deployment configurations for cloud platforms

---

## 🔄 **Workflow & File Dependencies**

### **Data Pipeline Flow:**
```
1. config.yaml → data_generation.py → data/raw/
2. data/raw/ → data_processing.py → data/processed/
3. data/processed/ → ml_models.py → data/models/
4. data/processed/ + data/models/ → dashboard.py → Web Interface
```

### **Key Configuration Files:**
- **`config.yaml`**: Central configuration for all components
- **`requirements.txt`**: Python package dependencies
- **`.gitignore`**: Version control exclusions

---

## 📈 **Project Organization Benefits**

### **✅ Professional Structure:**
- Clear separation of concerns
- Modular code organization
- Scalable architecture

### **✅ Easy Navigation:**
- Logical file grouping
- Descriptive naming conventions
- Comprehensive documentation

### **✅ Maintainability:**
- Isolated components
- Clear dependencies
- Version control ready

### **✅ Portfolio Ready:**
- Professional appearance
- Business documentation
- Technical depth

---

## 🚀 **Quick Start Commands**

### **Run Complete Pipeline:**
```bash
python3 scripts/run_pipeline.py
```

### **Run Individual Components:**
```bash
# Generate data
python3 src/data_generation.py

# Process data
python3 src/data_processing.py

# Train models
python3 src/ml_models.py

# Launch dashboard
streamlit run src/dashboard.py
```

### **Run Tests:**
```bash
python3 tests/test_dashboard.py
```

---

## 📝 **Future Enhancements**

### **Planned Additions:**
- Unit and integration tests
- CI/CD pipeline configuration
- Docker containerization
- Cloud deployment guides
- API documentation
- User guides and tutorials

### **Scalability Considerations:**
- Database integration
- Real-time data streaming
- Multi-tenant architecture
- Advanced ML models
- Mobile application

---

*This structure provides a solid foundation for a professional predictive maintenance dashboard project that can be easily maintained, extended, and showcased.* 