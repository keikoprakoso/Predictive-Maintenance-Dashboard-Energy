"""
Data Processing Module for Predictive Maintenance Dashboard
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    """Handles data cleaning, feature engineering, and preprocessing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.ml_config = self.config['ml_models']
        self.risk_config = self.config['risk_assessment']
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw sensor data"""
        print(f"📂 Loading data from: {filepath}")
        data = pd.read_csv(filepath)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        print(f"✅ Loaded {len(data):,} records")
        return data
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate sensor data"""
        print("🧹 Cleaning data...")
        
        initial_rows = len(data)
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"⚠️  Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Fill missing values with forward fill for time series
            data = data.sort_values(['turbine_id', 'timestamp'])
            data = data.fillna(method='ffill')
            
            # Fill any remaining NaNs with median
            numeric_columns = ['temperature_C', 'vibration_mm_s', 'runtime_hours']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())
        
        # Remove outliers using IQR method
        data = self._remove_outliers(data)
        
        # Validate data ranges
        data = self._validate_ranges(data)
        
        final_rows = len(data)
        print(f"✅ Cleaned data: {initial_rows:,} → {final_rows:,} records")
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numeric_columns = ['temperature_C', 'vibration_mm_s']
        
        for col in numeric_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"   Removed {outliers} outliers from {col}")
                
                # Remove outliers
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def _validate_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clip sensor values to reasonable ranges"""
        # Temperature validation (should be positive and reasonable for geothermal)
        data['temperature_C'] = np.clip(data['temperature_C'], 0, 200)
        
        # Vibration validation (should be positive)
        data['vibration_mm_s'] = np.clip(data['vibration_mm_s'], 0, 20)
        
        # Runtime validation (should be non-negative and reasonable)
        data['runtime_hours'] = np.clip(data['runtime_hours'], 0, 10000)
        
        return data
    
    def calculate_risk_assessment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive risk assessment for each record"""
        print("⚠️  Calculating risk assessment...")
        
        risk_data = data.copy()
        
        # Individual risk factors
        risk_data['temp_risk'] = np.where(
            risk_data['temperature_C'] > self.risk_config['temperature_critical'], 0.8,
            np.where(risk_data['temperature_C'] > self.risk_config['temperature_high_risk'], 0.5, 0.1)
        )
        
        risk_data['vib_risk'] = np.where(
            risk_data['vibration_mm_s'] > self.risk_config['vibration_critical'], 0.8,
            np.where(risk_data['vibration_mm_s'] > self.risk_config['vibration_high_risk'], 0.5, 0.1)
        )
        
        risk_data['runtime_risk'] = np.where(
            risk_data['runtime_hours'] > self.risk_config['runtime_high_risk'], 0.3, 0.05
        )
        
        # Trend-based risk (increasing values)
        risk_data['temp_trend_risk'] = np.where(
            risk_data['temp_rate_of_change'] > 2, 0.4,
            np.where(risk_data['temp_rate_of_change'] > 1, 0.2, 0.05)
        )
        
        risk_data['vib_trend_risk'] = np.where(
            risk_data['vib_rate_of_change'] > 0.5, 0.4,
            np.where(risk_data['vib_rate_of_change'] > 0.2, 0.2, 0.05)
        )
        
        # Combined risk score (weighted average)
        risk_data['comprehensive_risk_score'] = (
            risk_data['temp_risk'] * 0.35 +
            risk_data['vib_risk'] * 0.35 +
            risk_data['runtime_risk'] * 0.1 +
            risk_data['temp_trend_risk'] * 0.1 +
            risk_data['vib_trend_risk'] * 0.1
        )
        
        # Risk categories
        risk_data['risk_category'] = pd.cut(
            risk_data['comprehensive_risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        return risk_data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for ML models"""
        print("🔧 Engineering features...")
        
        # Sort by turbine and timestamp for rolling calculations
        data = data.sort_values(['turbine_id', 'timestamp']).reset_index(drop=True)
        
        # Time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['day_of_year'] = data['timestamp'].dt.dayofyear
        
        # Rolling averages for each turbine
        rolling_window = self.ml_config['rolling_window']
        
        for turbine_id in data['turbine_id'].unique():
            turbine_mask = data['turbine_id'] == turbine_id
            
            # Temperature rolling averages
            data.loc[turbine_mask, 'temp_rolling_mean_24h'] = (
                data.loc[turbine_mask, 'temperature_C'].rolling(window=rolling_window, min_periods=1).mean()
            )
            data.loc[turbine_mask, 'temp_rolling_std_24h'] = (
                data.loc[turbine_mask, 'temperature_C'].rolling(window=rolling_window, min_periods=1).std()
            )
            
            # Vibration rolling averages
            data.loc[turbine_mask, 'vib_rolling_mean_24h'] = (
                data.loc[turbine_mask, 'vibration_mm_s'].rolling(window=rolling_window, min_periods=1).mean()
            )
            data.loc[turbine_mask, 'vib_rolling_std_24h'] = (
                data.loc[turbine_mask, 'vibration_mm_s'].rolling(window=rolling_window, min_periods=1).std()
            )
        
        # Rate of change features
        data['temp_rate_of_change'] = data.groupby('turbine_id')['temperature_C'].diff()
        data['vib_rate_of_change'] = data.groupby('turbine_id')['vibration_mm_s'].diff()
        
        # Risk indicators
        data['temp_high_risk'] = (data['temperature_C'] > self.risk_config['temperature_high_risk']).astype(int)
        data['temp_critical'] = (data['temperature_C'] > self.risk_config['temperature_critical']).astype(int)
        data['vib_high_risk'] = (data['vibration_mm_s'] > self.risk_config['vibration_high_risk']).astype(int)
        data['vib_critical'] = (data['vibration_mm_s'] > self.risk_config['vibration_critical']).astype(int)
        data['runtime_high_risk'] = (data['runtime_hours'] > self.risk_config['runtime_high_risk']).astype(int)
        
        # Combined risk score
        data['risk_score'] = (
            data['temp_high_risk'] * 0.3 +
            data['temp_critical'] * 0.5 +
            data['vib_high_risk'] * 0.2 +
            data['vib_critical'] * 0.4 +
            data['runtime_high_risk'] * 0.1
        )
        
        # Failure lag features (for prediction)
        data['failure_lag_1h'] = data.groupby('turbine_id')['status'].shift(1)
        data['failure_lag_6h'] = data.groupby('turbine_id')['status'].shift(6)
        data['failure_lag_24h'] = data.groupby('turbine_id')['status'].shift(24)
        
        # Convert status to numeric
        data['status_numeric'] = (data['status'] == 'Fail').astype(int)
        
        print(f"✅ Added {len(data.columns) - 8} engineered features")
        return data
    
    def prepare_time_series_data(self, data: pd.DataFrame, turbine_id: int = None) -> pd.DataFrame:
        """Prepare data for time series forecasting"""
        if turbine_id:
            data = data[data['turbine_id'] == turbine_id].copy()
        
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Create Prophet-compatible format
        prophet_data = data[['timestamp', 'temperature_C', 'vibration_mm_s']].copy()
        prophet_data.columns = ['ds', 'temperature_C', 'vibration_mm_s']
        
        return prophet_data
    
    def prepare_ml_data(self, data: pd.DataFrame, target_column: str = 'status_numeric') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for traditional ML models"""
        # Select features for ML
        feature_columns = [
            'temperature_C', 'vibration_mm_s', 'runtime_hours',
            'hour', 'day_of_week', 'month', 'day_of_year',
            'temp_rolling_mean_24h', 'temp_rolling_std_24h',
            'vib_rolling_mean_24h', 'vib_rolling_std_24h',
            'temp_rate_of_change', 'vib_rate_of_change',
            'temp_high_risk', 'temp_critical', 'vib_high_risk', 
            'vib_critical', 'runtime_high_risk', 'risk_score'
        ]
        
        # Remove rows with NaN values
        data_clean = data[feature_columns + [target_column]].dropna()
        
        X = data_clean[feature_columns].values
        y = data_clean[target_column].values
        
        return X, y, feature_columns
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        # Sort by timestamp to maintain temporal order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate split indices
        total_size = len(data)
        test_split = int(total_size * (1 - test_size))
        val_split = int(test_split * (1 - val_size))
        
        train_data = data.iloc[:val_split]
        val_data = data.iloc[val_split:test_split]
        test_data = data.iloc[test_split:]
        
        print(f"📊 Data split:")
        print(f"   Train: {len(train_data):,} records ({len(train_data)/total_size:.1%})")
        print(f"   Validation: {len(val_data):,} records ({len(val_data)/total_size:.1%})")
        print(f"   Test: {len(test_data):,} records ({len(test_data)/total_size:.1%})")
        
        return train_data, val_data, test_data
    
    def get_latest_data(self, data: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
        """Get the most recent data for dashboard display"""
        latest_timestamp = data['timestamp'].max()
        cutoff_time = latest_timestamp - pd.Timedelta(hours=hours)
        
        recent_data = data[data['timestamp'] >= cutoff_time].copy()
        return recent_data
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate summary statistics for dashboard"""
        stats = {}
        
        # Overall statistics
        stats['total_records'] = len(data)
        stats['total_turbines'] = data['turbine_id'].nunique()
        stats['date_range'] = {
            'start': data['timestamp'].min().strftime('%Y-%m-%d'),
            'end': data['timestamp'].max().strftime('%Y-%m-%d')
        }
        
        # Failure statistics
        failure_data = data[data['status'] == 'Fail']
        stats['total_failures'] = len(failure_data)
        stats['failure_rate'] = len(failure_data) / len(data)
        
        # Sensor statistics
        stats['temperature'] = {
            'mean': data['temperature_C'].mean(),
            'std': data['temperature_C'].std(),
            'min': data['temperature_C'].min(),
            'max': data['temperature_C'].max(),
            'high_risk_count': len(data[data['temperature_C'] > self.risk_config['temperature_high_risk']])
        }
        
        stats['vibration'] = {
            'mean': data['vibration_mm_s'].mean(),
            'std': data['vibration_mm_s'].std(),
            'min': data['vibration_mm_s'].min(),
            'max': data['vibration_mm_s'].max(),
            'high_risk_count': len(data[data['vibration_mm_s'] > self.risk_config['vibration_high_risk']])
        }
        
        # Risk statistics
        stats['high_risk_records'] = len(data[data['risk_score'] > 0.5])
        stats['critical_records'] = len(data[data['risk_score'] > 0.8])
        
        return stats
    
    def save_processed_data(self, data: pd.DataFrame, filename: str = "processed_turbine_data.csv"):
        """Save processed data"""
        output_path = "data/processed/"
        os.makedirs(output_path, exist_ok=True)
        
        filepath = os.path.join(output_path, filename)
        data.to_csv(filepath, index=False)
        print(f"💾 Processed data saved to: {filepath}")
        
        return filepath

def main():
    """Main function to process sample data"""
    processor = DataProcessor()
    
    # Load raw data
    raw_data_path = "data/raw/turbine_sensor_data.csv"
    if not os.path.exists(raw_data_path):
        print("❌ Raw data not found. Please run data_generation.py first.")
        return
    
    data = processor.load_data(raw_data_path)
    
    # Clean data
    data = processor.clean_data(data)
    
    # Engineer features
    data = processor.engineer_features(data)
    
    # Calculate risk assessment (add comprehensive risk score)
    data = processor.calculate_risk_assessment(data)
    
    # Calculate statistics
    stats = processor.calculate_statistics(data)
    
    # Save processed data
    processor.save_processed_data(data)
    
    # Print summary
    print("\n📊 Processing Summary:")
    print(f"Total records: {stats['total_records']:,}")
    print(f"Turbines: {stats['total_turbines']}")
    print(f"Failures: {stats['total_failures']:,} ({stats['failure_rate']:.2%})")
    print(f"High risk records: {stats['high_risk_records']:,}")
    print(f"Critical records: {stats['critical_records']:,}")

if __name__ == "__main__":
    main() 