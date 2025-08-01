#!/usr/bin/env python3
"""
Test script to verify dashboard data availability
"""

import pandas as pd
import os

def test_dashboard_data():
    """Test if all required data files and columns are available"""
    print("🔍 Testing Dashboard Data Availability")
    print("=" * 50)
    
    # Check if processed data exists
    processed_path = "data/processed/processed_turbine_data.csv"
    if not os.path.exists(processed_path):
        print("❌ Processed data not found!")
        return False
    
    # Load processed data
    data = pd.read_csv(processed_path)
    print(f"✅ Processed data loaded: {len(data):,} records")
    
    # Check required columns
    required_columns = [
        'timestamp', 'turbine_id', 'temperature_C', 'vibration_mm_s', 
        'runtime_hours', 'status', 'comprehensive_risk_score', 'risk_category'
    ]
    
    missing_columns = []
    for col in required_columns:
        if col not in data.columns:
            missing_columns.append(col)
        else:
            print(f"✅ Column '{col}' available")
    
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False
    
    # Check predictions data
    predictions_path = "data/processed/predictions.csv"
    if os.path.exists(predictions_path):
        predictions = pd.read_csv(predictions_path)
        print(f"✅ Predictions data loaded: {len(predictions):,} records")
    else:
        print("⚠️  Predictions data not found")
    
    # Check recommendations data
    recommendations_path = "data/processed/maintenance_recommendations.csv"
    if os.path.exists(recommendations_path):
        recommendations = pd.read_csv(recommendations_path)
        print(f"✅ Recommendations data loaded: {len(recommendations):,} records")
    else:
        print("⚠️  Recommendations data not found")
    
    # Check models
    models_path = "data/models/"
    if os.path.exists(models_path):
        model_files = os.listdir(models_path)
        print(f"✅ Model files found: {len(model_files)} files")
        for file in model_files:
            print(f"   - {file}")
    else:
        print("⚠️  Models directory not found")
    
    # Data statistics
    print("\n📊 Data Statistics:")
    print(f"   Total records: {len(data):,}")
    print(f"   Turbines: {data['turbine_id'].nunique()}")
    print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"   Risk score range: {data['comprehensive_risk_score'].min():.3f} to {data['comprehensive_risk_score'].max():.3f}")
    
    # Risk distribution
    risk_dist = data['risk_category'].value_counts()
    print(f"   Risk distribution: {dict(risk_dist)}")
    
    print("\n✅ Dashboard data test completed successfully!")
    return True

if __name__ == "__main__":
    test_dashboard_data() 