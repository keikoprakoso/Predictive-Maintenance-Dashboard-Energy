"""
Machine Learning Models for Predictive Maintenance Dashboard
Implements time series forecasting, risk assessment, and failure prediction
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet for time series forecasting (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("✅ Prophet is available for time series forecasting")
except ImportError:
    print("⚠️  Prophet not available. Using alternative time series methods.")
    PROPHET_AVAILABLE = False

# Alternative time series methods
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("⚠️  Statsmodels not available for time series forecasting.")
    STATSMODELS_AVAILABLE = False

# Scikit-learn for traditional ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

class PredictiveMaintenanceModels:
    """Machine learning models for predictive maintenance"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.ml_config = self.config['ml_models']
        self.risk_config = self.config['risk_assessment']
        self.models_path = "data/models/"
        os.makedirs(self.models_path, exist_ok=True)
        
    def train_prophet_models(self, data: pd.DataFrame) -> Dict:
        """Train Prophet models for temperature and vibration forecasting"""
        if not PROPHET_AVAILABLE:
            print("❌ Prophet not available. Using alternative time series methods.")
            return self.train_alternative_models(data)
        
        print("🔮 Training Prophet models for time series forecasting...")
        
        models = {}
        
        # Train separate models for each turbine and sensor
        for turbine_id in data['turbine_id'].unique():
            turbine_data = data[data['turbine_id'] == turbine_id].copy()
            
            # Prepare data for Prophet
            prophet_data = turbine_data[['timestamp', 'temperature_C', 'vibration_mm_s']].copy()
            prophet_data.columns = ['ds', 'temperature_C', 'vibration_mm_s']
            
            # Train temperature model
            temp_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                interval_width=self.ml_config['confidence_interval']
            )
            
            temp_data = prophet_data[['ds', 'temperature_C']].copy()
            temp_data.columns = ['ds', 'y']
            temp_model.fit(temp_data)
            
            # Train vibration model
            vib_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                interval_width=self.ml_config['confidence_interval']
            )
            
            vib_data = prophet_data[['ds', 'vibration_mm_s']].copy()
            vib_data.columns = ['ds', 'y']
            vib_model.fit(vib_data)
            
            models[f'temp_turbine_{turbine_id}'] = temp_model
            models[f'vib_turbine_{turbine_id}'] = vib_model
            
            print(f"   ✅ Trained Prophet models for Turbine {turbine_id}")
        
        return models
    
    def train_alternative_models(self, data: pd.DataFrame) -> Dict:
        """Train alternative time series models when Prophet is not available"""
        if not STATSMODELS_AVAILABLE:
            print("❌ No time series forecasting libraries available. Skipping forecasting.")
            return {}
        
        print("🔮 Training alternative time series models...")
        
        models = {}
        
        # Train separate models for each turbine and sensor
        for turbine_id in data['turbine_id'].unique():
            turbine_data = data[data['turbine_id'] == turbine_id].copy()
            
            # Sort by timestamp
            turbine_data = turbine_data.sort_values('timestamp').reset_index(drop=True)
            
            # Train temperature model using Exponential Smoothing
            try:
                temp_model = ExponentialSmoothing(
                    turbine_data['temperature_C'],
                    seasonal_periods=24,  # Daily seasonality
                    trend='add',
                    seasonal='add'
                ).fit()
                models[f'temp_turbine_{turbine_id}'] = temp_model
                print(f"   ✅ Trained temperature model for Turbine {turbine_id}")
            except Exception as e:
                print(f"   ⚠️  Could not train temperature model for Turbine {turbine_id}: {e}")
            
            # Train vibration model using Exponential Smoothing
            try:
                vib_model = ExponentialSmoothing(
                    turbine_data['vibration_mm_s'],
                    seasonal_periods=24,  # Daily seasonality
                    trend='add',
                    seasonal='add'
                ).fit()
                models[f'vib_turbine_{turbine_id}'] = vib_model
                print(f"   ✅ Trained vibration model for Turbine {turbine_id}")
            except Exception as e:
                print(f"   ⚠️  Could not train vibration model for Turbine {turbine_id}: {e}")
        
        return models
    
    def forecast_sensor_values(self, models: Dict, 
                             periods: int = None) -> Dict[str, pd.DataFrame]:
        """Generate forecasts for sensor values"""
        if not models:
            return {}
        
        if periods is None:
            periods = self.ml_config['forecast_periods']
        
        print(f"🔮 Generating {periods}-day forecasts...")
        
        forecasts = {}
        
        for model_name, model in models.items():
            try:
                if PROPHET_AVAILABLE and hasattr(model, 'make_future_dataframe'):
                    # Prophet model
                    future = model.make_future_dataframe(periods=periods * 24, freq='H')
                    forecast = model.predict(future)
                    
                    # Keep only future predictions
                    latest_timestamp = model.history['ds'].max()
                    future_forecast = forecast[forecast['ds'] > latest_timestamp].copy()
                    
                elif STATSMODELS_AVAILABLE and hasattr(model, 'forecast'):
                    # Statsmodels model
                    forecast_values = model.forecast(steps=periods * 24)
                    
                    # Create future timestamps
                    last_timestamp = data[data['turbine_id'] == int(model_name.split('_')[-1])]['timestamp'].max()
                    future_timestamps = pd.date_range(
                        start=last_timestamp + pd.Timedelta(hours=1),
                        periods=periods * 24,
                        freq='H'
                    )
                    
                    future_forecast = pd.DataFrame({
                        'ds': future_timestamps,
                        'yhat': forecast_values,
                        'yhat_lower': forecast_values * 0.95,  # Simple confidence interval
                        'yhat_upper': forecast_values * 1.05
                    })
                
                else:
                    print(f"   ⚠️  Unknown model type for {model_name}")
                    continue
                
                forecasts[model_name] = future_forecast
                print(f"   ✅ Generated forecast for {model_name}")
                
            except Exception as e:
                print(f"   ⚠️  Error generating forecast for {model_name}: {e}")
        
        return forecasts
    
    def train_failure_prediction_model(self, data: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler, List[str]]:
        """Train Random Forest model for failure prediction"""
        print("🌲 Training Random Forest model for failure prediction...")
        
        # Prepare features and target
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
        data_clean = data[feature_columns + ['status_numeric']].dropna()
        
        X = data_clean[feature_columns].values
        y = data_clean['status_numeric'].values
        
        # Split data (maintaining temporal order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test_scaled)
        
        # Check if we have multiple classes
        if len(rf_model.classes_) > 1:
            y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
            print("📊 Model Performance:")
            print(f"   ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
            print(classification_report(y_test, y_pred))
        else:
            print("📊 Model Performance:")
            print("   Note: Only one class in training data (no failures detected)")
            print(f"   Accuracy: {rf_model.score(X_test_scaled, y_test):.3f}")
            print("   Model trained for future failure prediction")
        
        return rf_model, scaler, feature_columns
    
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
    
    def predict_maintenance_needs(self, data: pd.DataFrame, 
                                rf_model: RandomForestClassifier,
                                scaler: StandardScaler,
                                feature_columns: List[str]) -> pd.DataFrame:
        """Predict maintenance needs for each turbine"""
        print("🔮 Predicting maintenance needs...")
        
        prediction_data = data.copy()
        
        # Prepare features for prediction
        feature_data = prediction_data[feature_columns].fillna(0)
        X_scaled = scaler.transform(feature_data.values)
        
        # Get predictions
        if len(rf_model.classes_) > 1:
            failure_prob = rf_model.predict_proba(X_scaled)[:, 1]
        else:
            # If only one class, use risk score as probability
            failure_prob = prediction_data['comprehensive_risk_score'].values
        
        prediction_data['failure_probability'] = failure_prob
        
        # Determine maintenance urgency
        prediction_data['maintenance_urgency'] = pd.cut(
            failure_prob,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Calculate days until maintenance needed
        prediction_data['days_until_maintenance'] = np.where(
            failure_prob > 0.7, 7,
            np.where(failure_prob > 0.5, 30,
                    np.where(failure_prob > 0.3, 90, 365))
        )
        
        return prediction_data
    
    def calculate_cost_impact(self, data: pd.DataFrame) -> Dict:
        """Calculate cost impact of failures and maintenance"""
        print("💰 Calculating cost impact...")
        
        cost_config = self.config['costs']
        
        # Current failure statistics
        total_failures = len(data[data['status'] == 'Fail'])
        total_hours = len(data)
        
        # If no failures, create a hypothetical scenario for demonstration
        if total_failures == 0:
            # Assume 1 failure per turbine per year for demonstration
            total_failures = len(data['turbine_id'].unique())
            print(f"⚠️  No failures detected. Using hypothetical scenario: {total_failures} failures per year")
        
        # Calculate costs
        unplanned_maintenance_cost = total_failures * cost_config['unplanned_maintenance']
        downtime_cost = total_failures * 24 * cost_config['hourly_downtime_cost']  # Assume 24h downtime
        energy_loss_cost = total_failures * 24 * cost_config['energy_loss_per_hour']
        
        total_current_cost = unplanned_maintenance_cost + downtime_cost + energy_loss_cost
        
        # Calculate potential savings with predictive maintenance
        # Assume 70% of failures can be prevented with early maintenance
        preventable_failures = int(total_failures * 0.7)
        planned_maintenance_cost = preventable_failures * cost_config['planned_maintenance']
        
        # Reduced downtime (assume 4h instead of 24h)
        reduced_downtime_cost = preventable_failures * 4 * cost_config['hourly_downtime_cost']
        reduced_energy_loss = preventable_failures * 4 * cost_config['energy_loss_per_hour']
        
        total_predictive_cost = planned_maintenance_cost + reduced_downtime_cost + reduced_energy_loss
        
        # Calculate savings
        cost_savings = total_current_cost - total_predictive_cost
        roi_percentage = (cost_savings / total_predictive_cost) * 100 if total_predictive_cost > 0 else 0
        
        cost_impact = {
            'current_scenario': {
                'unplanned_maintenance_cost': unplanned_maintenance_cost,
                'downtime_cost': downtime_cost,
                'energy_loss_cost': energy_loss_cost,
                'total_cost': total_current_cost
            },
            'predictive_scenario': {
                'planned_maintenance_cost': planned_maintenance_cost,
                'reduced_downtime_cost': reduced_downtime_cost,
                'reduced_energy_loss': reduced_energy_loss,
                'total_cost': total_predictive_cost
            },
            'savings': {
                'total_savings': cost_savings,
                'roi_percentage': roi_percentage,
                'preventable_failures': preventable_failures
            }
        }
        
        return cost_impact
    
    def generate_maintenance_recommendations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate specific maintenance recommendations"""
        print("🔧 Generating maintenance recommendations...")
        
        recommendations = []
        
        for turbine_id in data['turbine_id'].unique():
            turbine_data = data[data['turbine_id'] == turbine_id].copy()
            latest_data = turbine_data.iloc[-1]
            
            # Determine recommendations based on current state
            recommendations_list = []
            priority = 'Low'
            
            # Temperature-based recommendations
            if latest_data['temperature_C'] > self.risk_config['temperature_critical']:
                recommendations_list.append("CRITICAL: Immediate shutdown required - temperature exceeds critical threshold")
                priority = 'Critical'
            elif latest_data['temperature_C'] > self.risk_config['temperature_high_risk']:
                recommendations_list.append("Schedule cooling system inspection and maintenance")
                priority = 'High'
            
            # Vibration-based recommendations
            if latest_data['vibration_mm_s'] > self.risk_config['vibration_critical']:
                recommendations_list.append("CRITICAL: Immediate shutdown required - vibration exceeds critical threshold")
                priority = 'Critical'
            elif latest_data['vibration_mm_s'] > self.risk_config['vibration_high_risk']:
                recommendations_list.append("Schedule bearing inspection and alignment check")
                priority = 'High'
            
            # Runtime-based recommendations
            if latest_data['runtime_hours'] > self.risk_config['runtime_high_risk']:
                recommendations_list.append("Schedule preventive maintenance - high runtime hours")
                priority = max(priority, 'Medium')
            
            # Risk score-based recommendations
            if latest_data['comprehensive_risk_score'] > 0.8:
                recommendations_list.append("High risk detected - schedule comprehensive inspection")
                priority = max(priority, 'High')
            elif latest_data['comprehensive_risk_score'] > 0.6:
                recommendations_list.append("Medium risk - monitor closely and schedule maintenance")
                priority = max(priority, 'Medium')
            
            # Default recommendation if no issues
            if not recommendations_list:
                recommendations_list.append("Continue normal operation - all parameters within normal range")
            
            recommendation = {
                'turbine_id': turbine_id,
                'timestamp': latest_data['timestamp'],
                'temperature_C': latest_data['temperature_C'],
                'vibration_mm_s': latest_data['vibration_mm_s'],
                'runtime_hours': latest_data['runtime_hours'],
                'risk_score': latest_data['comprehensive_risk_score'],
                'priority': priority,
                'recommendations': '; '.join(recommendations_list),
                'estimated_cost': self._estimate_maintenance_cost(priority)
            }
            
            recommendations.append(recommendation)
        
        return pd.DataFrame(recommendations)
    
    def _estimate_maintenance_cost(self, priority: str) -> float:
        """Estimate maintenance cost based on priority"""
        base_costs = {
            'Low': 5000,
            'Medium': 15000,
            'High': 35000,
            'Critical': 50000
        }
        return base_costs.get(priority, 10000)
    
    def save_models(self, models: Dict, rf_model: RandomForestClassifier, 
                   scaler: StandardScaler, feature_columns: List[str]):
        """Save trained models"""
        print("💾 Saving models...")
        
        # Save Prophet models
        for name, model in models.items():
            if PROPHET_AVAILABLE and hasattr(model, 'to_json'):
                model_path = os.path.join(self.models_path, f"{name}.json")
                with open(model_path, 'w') as f:
                    f.write(model.to_json())
            elif STATSMODELS_AVAILABLE and hasattr(model, 'save'):
                model_path = os.path.join(self.models_path, f"{name}.pkl")
                joblib.dump(model, model_path)
        
        # Save Random Forest model
        rf_path = os.path.join(self.models_path, "random_forest_model.pkl")
        joblib.dump(rf_model, rf_path)
        
        # Save scaler
        scaler_path = os.path.join(self.models_path, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        
        # Save feature columns
        features_path = os.path.join(self.models_path, "feature_columns.txt")
        with open(features_path, 'w') as f:
            f.write('\n'.join(feature_columns))
        
        print(f"✅ Models saved to {self.models_path}")
    
    def load_models(self) -> Tuple[Dict, RandomForestClassifier, StandardScaler, List[str]]:
        """Load trained models"""
        print("📂 Loading models...")
        
        models = {}
        
        # Load Prophet models
        for filename in os.listdir(self.models_path):
            if filename.endswith('.json'):
                model_name = filename.replace('.json', '')
                model_path = os.path.join(self.models_path, filename)
                
                if PROPHET_AVAILABLE:
                    with open(model_path, 'r') as f:
                        model_json = f.read()
                    model = Prophet()
                    model = model.from_json(model_json)
                    models[model_name] = model
        
        # Load alternative models
        for filename in os.listdir(self.models_path):
            if filename.endswith('.pkl') and not filename.startswith('random_forest') and not filename.startswith('scaler'):
                model_name = filename.replace('.pkl', '')
                model_path = os.path.join(self.models_path, filename)
                model = joblib.load(model_path)
                models[model_name] = model
        
        # Load Random Forest model
        rf_path = os.path.join(self.models_path, "random_forest_model.pkl")
        rf_model = joblib.load(rf_path)
        
        # Load scaler
        scaler_path = os.path.join(self.models_path, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        # Load feature columns
        features_path = os.path.join(self.models_path, "feature_columns.txt")
        with open(features_path, 'r') as f:
            feature_columns = f.read().splitlines()
        
        print(f"✅ Loaded {len(models)} models and Random Forest model")
        return models, rf_model, scaler, feature_columns

def main():
    """Main function to train and test models"""
    from data_processing import DataProcessor
    
    # Load and process data
    processor = DataProcessor()
    data_path = "data/processed/processed_turbine_data.csv"
    
    if not os.path.exists(data_path):
        print("❌ Processed data not found. Please run data_processing.py first.")
        return
    
    data = processor.load_data(data_path)
    
    # Initialize models
    ml_models = PredictiveMaintenanceModels()
    
    # Train models
    prophet_models = ml_models.train_prophet_models(data)
    rf_model, scaler, feature_columns = ml_models.train_failure_prediction_model(data)
    
    # Generate forecasts
    forecasts = ml_models.forecast_sensor_values(prophet_models)
    
    # Calculate risk assessment
    risk_data = ml_models.calculate_risk_assessment(data)
    
    # Predict maintenance needs
    prediction_data = ml_models.predict_maintenance_needs(
        risk_data, rf_model, scaler, feature_columns
    )
    
    # Calculate cost impact
    cost_impact = ml_models.calculate_cost_impact(data)
    
    # Generate recommendations
    recommendations = ml_models.generate_maintenance_recommendations(prediction_data)
    
    # Save models
    ml_models.save_models(prophet_models, rf_model, scaler, feature_columns)
    
    # Save results
    prediction_data.to_csv("data/processed/predictions.csv", index=False)
    recommendations.to_csv("data/processed/maintenance_recommendations.csv", index=False)
    
    # Print summary
    print("\n📊 Model Training Summary:")
    print(f"Time series models trained: {len(prophet_models)}")
    
    # Calculate Random Forest accuracy
    try:
        rf_accuracy = rf_model.score(scaler.transform(data[feature_columns].fillna(0).values), data['status_numeric'].fillna(0).values)
        print(f"Random Forest accuracy: {rf_accuracy:.3f}")
    except Exception as e:
        print(f"Random Forest evaluation: {e}")
    
    print(f"Cost savings potential: ${cost_impact['savings']['total_savings']:,.0f}")
    print(f"ROI: {cost_impact['savings']['roi_percentage']:.1f}%")

if __name__ == "__main__":
    main() 