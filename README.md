# Predictive Maintenance Dashboard (Energy Sector)

## Project Overview
A comprehensive predictive maintenance solution for geothermal energy companies, designed to predict equipment failures, optimize maintenance schedules, and reduce operational costs through data-driven insights.

**Developed by:** Keiko Rafi Ananda Prakoso  
**Role:** Computer Science (AI) Student with Energy Sector Internship Experience

## Business Objectives
- **Predict Equipment Failures:** Use ML to forecast potential turbine failures
- **Cost Optimization:** Compare planned vs unplanned maintenance costs
- **Operational Excellence:** Reduce downtime and improve efficiency
- **Data-Driven Decisions:** Provide actionable insights for stakeholders

## Project Architecture

```
├── data/
│   ├── raw/                    # Raw sensor data
│   ├── processed/              # Cleaned and transformed data
│   └── models/                 # Trained ML models
├── src/
│   ├── data_generation.py      # Mock data generation
│   ├── data_processing.py      # Data cleaning and feature engineering
│   ├── ml_models.py           # Prediction models
│   └── dashboard.py           # Streamlit dashboard
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── config.yaml
└── README.md
```

## Key Features

### 1. Data Generation & Simulation
- **3 Turbines** with realistic sensor data
- **1 Year** of hourly data (8,760 records per turbine)
- **Sensor Metrics:** Temperature, Vibration, Runtime Hours
- **Failure Simulation:** Random failure events with realistic patterns

### 2. Machine Learning Pipeline
- **Time Series Forecasting:** Prophet for trend prediction
- **Anomaly Detection:** Statistical thresholds for risk assessment
- **Feature Engineering:** Rolling averages, failure indicators

### 3. Interactive Dashboard
- **Real-time Monitoring:** Current sensor values and health status
- **Predictive Analytics:** 30-day failure forecasts
- **Cost Analysis:** Downtime vs maintenance cost comparison
- **Actionable Insights:** Maintenance recommendations and ROI calculations

### 4. Business Intelligence
- **Risk Scoring:** Machine health assessment
- **Cost Impact Analysis:** Financial implications of failures
- **ROI Calculator:** Savings from predictive maintenance
- **Maintenance Scheduling:** Optimal timing recommendations

## Dashboard Components

### Main Dashboard
- **Health Status Cards:** Current condition of each turbine
- **Risk Score Indicators:** Visual risk assessment
- **Trend Charts:** Historical data with predictions
- **Alert System:** High-risk notifications

### Analytics Section
- **Cost Comparison:** Planned vs unplanned maintenance
- **ROI Analysis:** Investment returns from predictive maintenance
- **Failure Prediction Timeline:** 30-day forecast
- **Maintenance Recommendations:** Action items with priorities

## Technology Stack

- **Python 3.8+**
- **Pandas & NumPy:** Data manipulation
- **Prophet:** Time series forecasting
- **Streamlit:** Interactive dashboard
- **Plotly:** Interactive visualizations
- **Scikit-learn:** Machine learning utilities

## Business Impact

### Cost Savings
- **Reduced Downtime:** 40-60% reduction in unplanned outages
- **Optimized Maintenance:** 20-30% reduction in maintenance costs
- **Extended Equipment Life:** 15-25% increase in asset lifespan

### Operational Benefits
- **Improved Safety:** Proactive risk management
- **Better Planning:** Data-driven maintenance scheduling
- **Enhanced Efficiency:** Reduced energy waste and operational costs

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Complete Pipeline:**
   ```bash
   python scripts/run_pipeline.py
   ```

3. **Run Dashboard:**
   ```bash
   streamlit run src/dashboard.py
   ```

### 📁 Project Structure
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization and file descriptions.

## Data Storytelling

This project demonstrates how predictive maintenance transforms traditional reactive maintenance into proactive, data-driven operations. By leveraging sensor data and machine learning, energy companies can:

- **Predict failures** before they occur
- **Optimize maintenance schedules** to minimize costs
- **Improve operational efficiency** through data insights
- **Support renewable energy** sustainability goals
