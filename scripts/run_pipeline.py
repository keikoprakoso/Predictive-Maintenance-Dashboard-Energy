#!/usr/bin/env python3
"""
Predictive Maintenance Dashboard - Complete Pipeline
Orchestrates the entire data generation, processing, model training, and dashboard setup
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'streamlit', 
        'plotly', 'pyyaml', 'tqdm', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def run_data_generation():
    """Run data generation step"""
    print_step(1, "Generating Mock Turbine Data")
    
    try:
        from src.data_generation import TurbineDataGenerator
        
        generator = TurbineDataGenerator()
        data = generator.generate_sample_data()
        
        print(f"✅ Generated {len(data):,} records")
        print(f"   Turbines: {data['turbine_id'].nunique()}")
        print(f"   Failures: {len(data[data['status'] == 'Fail']):,}")
        
        return True
    except Exception as e:
        print(f"❌ Error in data generation: {str(e)}")
        return False

def run_data_processing():
    """Run data processing step"""
    print_step(2, "Processing and Feature Engineering")
    
    try:
        from src.data_processing import DataProcessor
        
        processor = DataProcessor()
        
        # Load raw data
        raw_data_path = "data/raw/turbine_sensor_data.csv"
        if not os.path.exists(raw_data_path):
            print("❌ Raw data not found. Please run data generation first.")
            return False
        
        data = processor.load_data(raw_data_path)
        
        # Clean and process data
        data = processor.clean_data(data)
        data = processor.engineer_features(data)
        
        # Save processed data
        processor.save_processed_data(data)
        
        # Calculate statistics
        stats = processor.calculate_statistics(data)
        
        print(f"✅ Processed {stats['total_records']:,} records")
        print(f"   Features: {len(data.columns)}")
        print(f"   Failures: {stats['total_failures']:,} ({stats['failure_rate']:.2%})")
        
        return True
    except Exception as e:
        print(f"❌ Error in data processing: {str(e)}")
        return False

def run_model_training():
    """Run model training step"""
    print_step(3, "Training Machine Learning Models")
    
    try:
        from src.ml_models import PredictiveMaintenanceModels
        
        ml_models = PredictiveMaintenanceModels()
        
        # Load processed data
        processed_data_path = "data/processed/processed_turbine_data.csv"
        if not os.path.exists(processed_data_path):
            print("❌ Processed data not found. Please run data processing first.")
            return False
        
        from src.data_processing import DataProcessor
        processor = DataProcessor()
        data = processor.load_data(processed_data_path)
        
        # Train models
        print("   Training Prophet models...")
        prophet_models = ml_models.train_prophet_models(data)
        
        print("   Training Random Forest model...")
        rf_model, scaler, feature_columns = ml_models.train_failure_prediction_model(data)
        
        # Generate predictions and analysis
        print("   Calculating risk assessment...")
        risk_data = ml_models.calculate_risk_assessment(data)
        
        print("   Generating predictions...")
        predictions = ml_models.predict_maintenance_needs(
            risk_data, rf_model, scaler, feature_columns
        )
        
        print("   Calculating cost impact...")
        cost_impact = ml_models.calculate_cost_impact(data)
        
        print("   Generating recommendations...")
        recommendations = ml_models.generate_maintenance_recommendations(predictions)
        
        # Save models and results
        ml_models.save_models(prophet_models, rf_model, scaler, feature_columns)
        
        # Save results
        predictions.to_csv("data/processed/predictions.csv", index=False)
        recommendations.to_csv("data/processed/maintenance_recommendations.csv", index=False)
        
        print(f"✅ Trained {len(prophet_models)} Prophet models and 1 Random Forest model")
        print(f"   Cost savings potential: ${cost_impact['savings']['total_savings']:,.0f}")
        print(f"   ROI: {cost_impact['savings']['roi_percentage']:.1f}%")
        print(f"   Recommendations: {len(recommendations)}")
        
        return True
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    print_step(4, "Launching Dashboard")
    
    try:
        # Check if dashboard file exists
        dashboard_path = "src/dashboard.py"
        if not os.path.exists(dashboard_path):
            print("❌ Dashboard file not found.")
            return False
        
        print("🌐 Starting Streamlit dashboard...")
        print("   Dashboard will open in your browser automatically.")
        print("   If it doesn't open, go to: http://localhost:8501")
        print("   Press Ctrl+C to stop the dashboard.")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path, "--server.port", "8501"
        ])
        
        return True
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user.")
        return True
    except Exception as e:
        print(f"❌ Error running dashboard: {str(e)}")
        return False

def show_summary():
    """Show pipeline summary"""
    print_header("Pipeline Summary")
    
    # Check generated files
    files_to_check = [
        ("data/raw/turbine_sensor_data.csv", "Raw sensor data"),
        ("data/processed/processed_turbine_data.csv", "Processed data"),
        ("data/models/random_forest_model.pkl", "Random Forest model"),
        ("data/processed/predictions.csv", "Predictions"),
        ("data/processed/maintenance_recommendations.csv", "Recommendations")
    ]
    
    print("📁 Generated Files:")
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   ✅ {description}: {filepath} ({size:.1f} KB)")
        else:
            print(f"   ❌ {description}: {filepath} - Missing")
    
    print("\n🎯 Next Steps:")
    print("   1. Review the dashboard at http://localhost:8501")
    print("   2. Explore the Jupyter notebook: notebooks/exploratory_analysis.ipynb")
    print("   3. Customize the configuration in config.yaml")
    print("   4. Deploy to production environment")

def main():
    """Main pipeline execution"""
    print_header("Predictive Maintenance Dashboard Pipeline")
    print("Developed by: Keiko Rafi Ananda Prakoso")
    print("Energy Sector AI Solutions")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    # Run pipeline steps
    steps = [
        ("Data Generation", run_data_generation),
        ("Data Processing", run_data_processing),
        ("Model Training", run_model_training)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Pipeline failed at: {step_name}")
            return
    
    # Show summary
    show_summary()
    
    # Ask if user wants to run dashboard
    print("\n" + "="*60)
    response = input("🚀 Would you like to launch the dashboard now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_dashboard()
    else:
        print("\n💡 To run the dashboard later, use:")
        print("   streamlit run src/dashboard.py")
        print("\n📚 To explore the data, open:")
        print("   notebooks/exploratory_analysis.ipynb")

if __name__ == "__main__":
    main() 