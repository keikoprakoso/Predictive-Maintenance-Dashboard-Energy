"""
Data Generation Module for Predictive Maintenance Dashboard
Simulates realistic turbine sensor data with failures and degradation patterns
"""

import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random

class TurbineDataGenerator:
    """Generates realistic turbine sensor data with failure simulation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_config = self.config['data_generation']
        self.sensor_config = self.config['sensors']
        self.failure_config = self.config['failure_simulation']
        
    def generate_timestamps(self) -> pd.DatetimeIndex:
        """Generate hourly timestamps for the entire year"""
        start_date = datetime.strptime(self.data_config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.data_config['end_date'], '%Y-%m-%d')
        
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=self.data_config['frequency']
        )
        return timestamps
    
    def generate_normal_temperature(self, hours: int, base_temp: float = 65) -> np.ndarray:
        """Generate normal temperature data with seasonal and daily patterns"""
        # Seasonal pattern (higher in summer, lower in winter)
        seasonal_pattern = np.sin(2 * np.pi * np.arange(hours) / (24 * 365)) * 10
        
        # Daily pattern (higher during day, lower at night)
        daily_pattern = np.sin(2 * np.pi * np.arange(hours) / 24) * 5
        
        # Random noise
        noise = np.random.normal(0, self.sensor_config['temperature']['noise_std'], hours)
        
        # Combine patterns
        temperature = base_temp + seasonal_pattern + daily_pattern + noise
        
        # Ensure within normal range
        temp_range = self.sensor_config['temperature']['normal_range']
        temperature = np.clip(temperature, temp_range[0], temp_range[1])
        
        return temperature
    
    def generate_normal_vibration(self, hours: int, base_vibration: float = 1.5) -> np.ndarray:
        """Generate normal vibration data with operational patterns"""
        # Operational pattern (higher during peak hours)
        operational_pattern = np.sin(2 * np.pi * np.arange(hours) / 24) * 0.5
        
        # Gradual increase over time (wear and tear)
        wear_pattern = np.linspace(0, 0.3, hours)
        
        # Random noise
        noise = np.random.normal(0, self.sensor_config['vibration']['noise_std'], hours)
        
        # Combine patterns
        vibration = base_vibration + operational_pattern + wear_pattern + noise
        
        # Ensure within normal range
        vib_range = self.sensor_config['vibration']['normal_range']
        vibration = np.clip(vibration, vib_range[0], vib_range[1])
        
        return vibration
    
    def generate_runtime_hours(self, hours: int) -> np.ndarray:
        """Generate cumulative runtime hours"""
        return np.arange(hours)
    
    def simulate_failures(self, hours: int) -> Tuple[List[Dict], np.ndarray]:
        """Simulate failure events and their impact on sensor data"""
        failures = []
        failure_indicators = np.zeros(hours)
        
        # Generate random failure events
        failure_prob = self.failure_config['failure_probability'] / 24  # Convert to hourly
        
        for hour in range(hours):
            if random.random() < failure_prob:
                # Determine failure duration
                duration = random.randint(
                    self.failure_config['failure_duration_hours'][0],
                    self.failure_config['failure_duration_hours'][1]
                )
                
                # Create failure event
                failure = {
                    'start_hour': hour,
                    'end_hour': min(hour + duration, hours),
                    'duration': duration,
                    'type': random.choice(['temperature', 'vibration', 'combined'])
                }
                failures.append(failure)
                
                # Mark failure period
                failure_indicators[hour:min(hour + duration, hours)] = 1
        
        return failures, failure_indicators
    
    def apply_failure_effects(self, data: np.ndarray, failures: List[Dict], 
                            sensor_type: str, hours: int) -> np.ndarray:
        """Apply failure effects to sensor data"""
        modified_data = data.copy()
        
        for failure in failures:
            start_hour = failure['start_hour']
            end_hour = failure['end_hour']
            
            if failure['type'] in [sensor_type, 'combined']:
                # Degradation period (gradual increase before failure)
                degradation_start = max(0, start_hour - self.failure_config['degradation_period_hours'])
                
                if sensor_type == 'temperature':
                    # Gradual temperature increase
                    for i in range(degradation_start, end_hour):
                        if i < start_hour:
                            # Degradation phase
                            progress = (i - degradation_start) / self.failure_config['degradation_period_hours']
                            increase = progress * 20  # Gradual increase
                        else:
                            # Failure phase
                            increase = 20 + random.uniform(10, 30)
                        
                        modified_data[i] = min(
                            modified_data[i] + increase,
                            self.sensor_config['temperature']['failure_threshold'] + 10
                        )
                
                elif sensor_type == 'vibration':
                    # Gradual vibration increase
                    for i in range(degradation_start, end_hour):
                        if i < start_hour:
                            # Degradation phase
                            progress = (i - degradation_start) / self.failure_config['degradation_period_hours']
                            increase = progress * 2
                        else:
                            # Failure phase
                            increase = 2 + random.uniform(1, 3)
                        
                        modified_data[i] = min(
                            modified_data[i] + increase,
                            self.sensor_config['vibration']['failure_threshold'] + 2
                        )
        
        return modified_data
    
    def generate_turbine_data(self, turbine_id: int, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate complete dataset for a single turbine"""
        hours = len(timestamps)
        
        # Generate base sensor data
        temperature = self.generate_normal_temperature(hours, base_temp=60 + turbine_id * 5)
        vibration = self.generate_normal_vibration(hours, base_vibration=1.2 + turbine_id * 0.3)
        runtime_hours = self.generate_runtime_hours(hours)
        
        # Simulate failures
        failures, failure_indicators = self.simulate_failures(hours)
        
        # Apply failure effects
        temperature = self.apply_failure_effects(temperature, failures, 'temperature', hours)
        vibration = self.apply_failure_effects(vibration, failures, 'vibration', hours)
        
        # Determine status based on sensor values and failure indicators
        status = []
        for i in range(hours):
            temp_threshold = self.sensor_config['temperature']['failure_threshold']
            vib_threshold = self.sensor_config['vibration']['failure_threshold']
            
            if (temperature[i] >= temp_threshold or 
                vibration[i] >= vib_threshold or 
                failure_indicators[i] == 1):
                status.append('Fail')
            else:
                status.append('Normal')
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'turbine_id': turbine_id,
            'temperature_C': temperature,
            'vibration_mm_s': vibration,
            'runtime_hours': runtime_hours,
            'status': status,
            'failure_indicator': failure_indicators
        })
        
        return df
    
    def generate_all_data(self) -> pd.DataFrame:
        """Generate data for all turbines"""
        print("🔄 Generating turbine sensor data...")
        
        timestamps = self.generate_timestamps()
        all_data = []
        
        for turbine_id in range(1, self.data_config['num_turbines'] + 1):
            print(f"   Generating data for Turbine {turbine_id}...")
            turbine_data = self.generate_turbine_data(turbine_id, timestamps)
            all_data.append(turbine_data)
        
        # Combine all turbine data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp and turbine_id
        combined_data = combined_data.sort_values(['timestamp', 'turbine_id']).reset_index(drop=True)
        
        print(f"✅ Generated {len(combined_data)} records for {self.data_config['num_turbines']} turbines")
        print(f"   Date range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
        
        return combined_data
    
    def save_data(self, data: pd.DataFrame, filename: str = "turbine_sensor_data.csv"):
        """Save generated data to file"""
        output_path = self.data_config['output_path']
        os.makedirs(output_path, exist_ok=True)
        
        filepath = os.path.join(output_path, filename)
        data.to_csv(filepath, index=False)
        print(f"💾 Data saved to: {filepath}")
        
        return filepath
    
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate and save sample data"""
        data = self.generate_all_data()
        self.save_data(data)
        return data

def main():
    """Main function to generate sample data"""
    generator = TurbineDataGenerator()
    data = generator.generate_sample_data()
    
    # Print summary statistics
    print("\n📊 Data Summary:")
    print(f"Total records: {len(data):,}")
    print(f"Turbines: {data['turbine_id'].nunique()}")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Failures: {len(data[data['status'] == 'Fail']):,}")
    
    # Print sensor statistics
    print("\n🌡️ Temperature Statistics:")
    print(data['temperature_C'].describe())
    
    print("\n📳 Vibration Statistics:")
    print(data['vibration_mm_s'].describe())
    
    print("\n⏱️ Runtime Statistics:")
    print(data['runtime_hours'].describe())

if __name__ == "__main__":
    main() 