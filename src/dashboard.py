"""
Predictive Maintenance Dashboard
Interactive Streamlit dashboard for geothermal turbine monitoring and maintenance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_processing import DataProcessor
from ml_models import PredictiveMaintenanceModels

# Page configuration
st.set_page_config(
    page_title="🏭 Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #d32f2f;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        color: #e65100;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    /* Improve text visibility */
    .stMarkdown {
        color: #212529;
    }
    .stText {
        color: #212529;
    }
    /* Better contrast for metrics */
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class PredictiveMaintenanceDashboard:
    """Main dashboard class for predictive maintenance"""
    
    def __init__(self):
        """Initialize dashboard components"""
        self.config = self.load_config()
        self.data_processor = DataProcessor()
        self.ml_models = PredictiveMaintenanceModels()
        
        # Load data
        self.data = self.load_data()
        self.recommendations = self.load_recommendations()
        
    def load_config(self):
        """Load configuration"""
        with open("config.yaml", 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self):
        """Load processed data"""
        data_path = "data/processed/processed_turbine_data.csv"
        if os.path.exists(data_path):
            return self.data_processor.load_data(data_path)
        else:
            st.error("❌ Processed data not found. Please run the data processing pipeline first.")
            return None
    
    def load_recommendations(self):
        """Load maintenance recommendations"""
        rec_path = "data/processed/maintenance_recommendations.csv"
        if os.path.exists(rec_path):
            return pd.read_csv(rec_path)
        return pd.DataFrame()
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🏭 Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("**Geothermal Energy Sector | Real-time Turbine Monitoring & Predictive Analytics**")
        st.markdown("---")
    
    def render_overview_metrics(self):
        """Render overview metrics cards"""
        if self.data is None:
            return
        
        st.subheader("📊 System Overview")
        
        # Calculate key metrics
        total_turbines = self.data['turbine_id'].nunique()
        total_records = len(self.data)
        total_failures = len(self.data[self.data['status'] == 'Fail'])
        failure_rate = total_failures / total_records if total_records > 0 else 0
        
        # Current status
        latest_data = self.data.groupby('turbine_id').last().reset_index()
        operational_turbines = len(latest_data[latest_data['status'] == 'Normal'])
        failed_turbines = len(latest_data[latest_data['status'] == 'Fail'])
        
        # High risk turbines
        high_risk_turbines = len(latest_data[latest_data['comprehensive_risk_score'] > 0.6])
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔄 Total Turbines",
                value=total_turbines,
                help="Number of turbines in the system"
            )
        
        with col2:
            st.metric(
                label="✅ Operational",
                value=operational_turbines,
                delta=f"{operational_turbines}/{total_turbines}",
                help="Currently operational turbines"
            )
        
        with col3:
            st.metric(
                label="⚠️ High Risk",
                value=high_risk_turbines,
                delta=f"{high_risk_turbines}/{total_turbines}",
                help="Turbines with high risk scores"
            )
        
        with col4:
            st.metric(
                label="📈 Failure Rate",
                value=f"{failure_rate:.2%}",
                help="Overall system failure rate"
            )
    
    def render_turbine_status(self):
        """Render individual turbine status"""
        if self.data is None:
            return
        
        st.subheader("🔍 Turbine Status")
        
        # Get latest data for each turbine
        latest_data = self.data.groupby('turbine_id').last().reset_index()
        
        # Create status cards for each turbine
        cols = st.columns(len(latest_data))
        
        for idx, (_, turbine) in enumerate(latest_data.iterrows()):
            with cols[idx]:
                # Determine alert class based on risk score
                risk_score = turbine['comprehensive_risk_score']
                if risk_score > 0.7:
                    alert_class = "alert-high"
                elif risk_score > 0.4:
                    alert_class = "alert-medium"
                else:
                    alert_class = "alert-low"
                
                # Create status card
                st.markdown(f"""
                <div class="metric-card {alert_class}">
                    <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">Turbine {turbine['turbine_id']}</h3>
                    <p style="margin: 0.25rem 0;"><strong>Status:</strong> <span style="color: {'#d32f2f' if turbine['status'] == 'Fail' else '#2e7d32'}">{turbine['status']}</span></p>
                    <p style="margin: 0.25rem 0;"><strong>Temperature:</strong> {turbine['temperature_C']:.1f}°C</p>
                    <p style="margin: 0.25rem 0;"><strong>Vibration:</strong> {turbine['vibration_mm_s']:.2f} mm/s</p>
                    <p style="margin: 0.25rem 0;"><strong>Runtime:</strong> {turbine['runtime_hours']:.0f} hours</p>
                    <p style="margin: 0.25rem 0;"><strong>Risk Score:</strong> <span style="color: {'#d32f2f' if risk_score > 0.6 else '#e65100' if risk_score > 0.3 else '#2e7d32'}">{risk_score:.2f}</span></p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_sensor_trends(self):
        """Render sensor trends with predictions"""
        if self.data is None:
            return
        
        st.subheader("📈 Sensor Trends & Predictions")
        
        # Turbine selector
        turbine_id = st.selectbox(
            "Select Turbine:",
            options=sorted(self.data['turbine_id'].unique()),
            key="turbine_selector"
        )
        
        # Filter data for selected turbine
        turbine_data = self.data[self.data['turbine_id'] == turbine_id].copy()
        
        # Create subplots for temperature and vibration
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Trend (°C)', 'Vibration Trend (mm/s)'),
            vertical_spacing=0.1
        )
        
        # Temperature plot
        fig.add_trace(
            go.Scatter(
                x=turbine_data['timestamp'],
                y=turbine_data['temperature_C'],
                mode='lines',
                name='Temperature',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Add temperature thresholds
        temp_high = self.config['risk_assessment']['temperature_high_risk']
        temp_critical = self.config['risk_assessment']['temperature_critical']
        
        fig.add_hline(
            y=temp_high, line_dash="dash", line_color="orange",
            annotation_text="High Risk Threshold", row=1, col=1
        )
        fig.add_hline(
            y=temp_critical, line_dash="dash", line_color="red",
            annotation_text="Critical Threshold", row=1, col=1
        )
        
        # Vibration plot
        fig.add_trace(
            go.Scatter(
                x=turbine_data['timestamp'],
                y=turbine_data['vibration_mm_s'],
                mode='lines',
                name='Vibration',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Add vibration thresholds
        vib_high = self.config['risk_assessment']['vibration_high_risk']
        vib_critical = self.config['risk_assessment']['vibration_critical']
        
        fig.add_hline(
            y=vib_high, line_dash="dash", line_color="orange",
            annotation_text="High Risk Threshold", row=2, col=1
        )
        fig.add_hline(
            y=vib_critical, line_dash="dash", line_color="red",
            annotation_text="Critical Threshold", row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title=f"Turbine {turbine_id} - Sensor Trends"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analysis(self):
        """Render risk analysis and predictions"""
        if self.data is None:
            return
        
        st.subheader("⚠️ Risk Analysis & Predictions")
        
        # Get latest predictions
        predictions_path = "data/processed/predictions.csv"
        if os.path.exists(predictions_path):
            try:
                predictions = pd.read_csv(predictions_path)
                predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
                
                # Filter for recent data
                latest_predictions = predictions.groupby('turbine_id').last().reset_index()
                
                # Ensure required columns exist
                required_cols = ['temperature_C', 'vibration_mm_s', 'comprehensive_risk_score', 'maintenance_urgency']
                missing_cols = [col for col in required_cols if col not in latest_predictions.columns]
                
                if missing_cols:
                    st.warning(f"Missing columns in predictions data: {missing_cols}")
                    return
                
                # Create risk visualization
                # Ensure data types are correct
                plot_data = latest_predictions.copy()
                plot_data['temperature_C'] = pd.to_numeric(plot_data['temperature_C'], errors='coerce')
                plot_data['vibration_mm_s'] = pd.to_numeric(plot_data['vibration_mm_s'], errors='coerce')
                plot_data['comprehensive_risk_score'] = pd.to_numeric(plot_data['comprehensive_risk_score'], errors='coerce')
                
                # Remove any NaN values
                plot_data = plot_data.dropna(subset=['temperature_C', 'vibration_mm_s', 'comprehensive_risk_score'])
                
                if len(plot_data) > 0:
                    # Ensure hover_data columns exist
                    hover_columns = []
                    for col in ['turbine_id', 'runtime_hours', 'failure_probability']:
                        if col in plot_data.columns:
                            hover_columns.append(col)
                    
                    fig = px.scatter(
                        plot_data,
                        x='temperature_C',
                        y='vibration_mm_s',
                        size='comprehensive_risk_score',
                        color='maintenance_urgency',
                        hover_data=hover_columns,
                        title="Turbine Risk Assessment Matrix",
                        color_discrete_map={
                            'Low': '#2E8B57',  # Sea Green
                            'Medium': '#FF8C00',  # Dark Orange
                            'High': '#DC143C'  # Crimson
                        }
                    )
                else:
                    # Create empty plot if no data
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No risk data available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    fig.update_layout(title="Turbine Risk Assessment Matrix")
                
                # Add threshold lines
                temp_high = self.config['risk_assessment']['temperature_high_risk']
                vib_high = self.config['risk_assessment']['vibration_high_risk']
                
                fig.add_vline(x=temp_high, line_dash="dash", line_color="orange")
                fig.add_hline(y=vib_high, line_dash="dash", line_color="orange")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk summary table
                st.subheader("Risk Summary")
                risk_summary = latest_predictions[['turbine_id', 'comprehensive_risk_score', 
                                                'maintenance_urgency', 'failure_probability', 
                                                'days_until_maintenance']].copy()
                risk_summary['failure_probability'] = risk_summary['failure_probability'].apply(lambda x: f"{x:.1%}")
                risk_summary['comprehensive_risk_score'] = risk_summary['comprehensive_risk_score'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(risk_summary, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading predictions data: {str(e)}")
                st.info("Please ensure the ML models have been trained successfully.")
    
    def render_cost_analysis(self):
        """Render cost analysis and ROI"""
        if self.data is None:
            return
        
        st.subheader("💰 Cost Analysis & ROI")
        
        # Calculate cost impact
        cost_impact = self.ml_models.calculate_cost_impact(self.data)
        
        # Create cost comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Scenario (Reactive Maintenance)")
            current = cost_impact['current_scenario']
            
            st.metric("Unplanned Maintenance", f"${current['unplanned_maintenance_cost']:,.0f}")
            st.metric("Downtime Cost", f"${current['downtime_cost']:,.0f}")
            st.metric("Energy Loss", f"${current['energy_loss_cost']:,.0f}")
            st.metric("**Total Cost**", f"**${current['total_cost']:,.0f}**", delta="Baseline")
        
        with col2:
            st.markdown("### Predictive Maintenance Scenario")
            predictive = cost_impact['predictive_scenario']
            savings = cost_impact['savings']
            
            st.metric("Planned Maintenance", f"${predictive['planned_maintenance_cost']:,.0f}")
            st.metric("Reduced Downtime", f"${predictive['reduced_downtime_cost']:,.0f}")
            st.metric("Reduced Energy Loss", f"${predictive['reduced_energy_loss']:,.0f}")
            st.metric("**Total Cost**", f"**${predictive['total_cost']:,.0f}**", 
                     delta=f"-${savings['total_savings']:,.0f}")
        
        # ROI Analysis
        st.markdown("### ROI Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Savings", f"${savings['total_savings']:,.0f}")
        
        with col2:
            st.metric("ROI", f"{savings['roi_percentage']:.1f}%")
        
        with col3:
            st.metric("Preventable Failures", savings['preventable_failures'])
        
        # Cost savings visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current Cost',
            x=['Unplanned Maintenance', 'Downtime Cost', 'Energy Loss'],
            y=[current['unplanned_maintenance_cost'], current['downtime_cost'], current['energy_loss_cost']],
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            name='Predictive Cost',
            x=['Planned Maintenance', 'Reduced Downtime', 'Reduced Energy Loss'],
            y=[predictive['planned_maintenance_cost'], predictive['reduced_downtime_cost'], predictive['reduced_energy_loss']],
            marker_color='green'
        ))
        
        fig.update_layout(
            title="Cost Comparison: Current vs Predictive Maintenance",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_maintenance_recommendations(self):
        """Render maintenance recommendations"""
        if self.recommendations.empty:
            st.warning("No maintenance recommendations available.")
            return
        
        st.subheader("🔧 Maintenance Recommendations")
        
        # Priority filter
        priority_filter = st.selectbox(
            "Filter by Priority:",
            options=['All'] + list(self.recommendations['priority'].unique()),
            key="priority_filter"
        )
        
        # Filter recommendations
        if priority_filter != 'All':
            filtered_recs = self.recommendations[self.recommendations['priority'] == priority_filter]
        else:
            filtered_recs = self.recommendations
        
        # Display recommendations
        for _, rec in filtered_recs.iterrows():
            # Determine color based on priority
            if rec['priority'] == 'Critical':
                color = "#f44336"
            elif rec['priority'] == 'High':
                color = "#ff9800"
            elif rec['priority'] == 'Medium':
                color = "#ffc107"
            else:
                color = "#4caf50"
            
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 1rem; margin: 1rem 0; background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">Turbine {rec['turbine_id']} - {rec['priority']} Priority</h4>
                <p style="margin: 0.25rem 0; color: #212529;"><strong>Risk Score:</strong> <span style="color: {'#d32f2f' if rec['risk_score'] > 0.6 else '#e65100' if rec['risk_score'] > 0.3 else '#2e7d32'}">{rec['risk_score']:.2f}</span></p>
                <p style="margin: 0.25rem 0; color: #212529;"><strong>Current Values:</strong> Temp: {rec['temperature_C']:.1f}°C, Vib: {rec['vibration_mm_s']:.2f} mm/s</p>
                <p style="margin: 0.25rem 0; color: #212529;"><strong>Recommendation:</strong> {rec['recommendations']}</p>
                <p style="margin: 0.25rem 0; color: #212529;"><strong>Estimated Cost:</strong> ${rec['estimated_cost']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_data_insights(self):
        """Render data insights and patterns"""
        if self.data is None:
            return
        
        st.subheader("📊 Data Insights & Patterns")
        
        # Time-based analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Failure distribution by hour
            failure_hourly = self.data[self.data['status'] == 'Fail'].groupby('hour').size()
            
            fig = px.bar(
                x=failure_hourly.index,
                y=failure_hourly.values,
                title="Failure Distribution by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Failures'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Failure distribution by month
            failure_monthly = self.data[self.data['status'] == 'Fail'].groupby('month').size()
            
            fig = px.bar(
                x=failure_monthly.index,
                y=failure_monthly.values,
                title="Failure Distribution by Month",
                labels={'x': 'Month', 'y': 'Number of Failures'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Sensor Correlation Analysis")
        
        correlation_data = self.data[['temperature_C', 'vibration_mm_s', 'runtime_hours']].corr()
        
        fig = px.imshow(
            correlation_data,
            title="Sensor Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run_dashboard(self):
        """Run the main dashboard"""
        self.render_header()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Overview", 
            "📈 Trends", 
            "⚠️ Risk Analysis", 
            "💰 Cost Analysis", 
            "🔧 Maintenance", 
            "📊 Insights"
        ])
        
        with tab1:
            self.render_overview_metrics()
            self.render_turbine_status()
        
        with tab2:
            self.render_sensor_trends()
        
        with tab3:
            self.render_risk_analysis()
        
        with tab4:
            self.render_cost_analysis()
        
        with tab5:
            self.render_maintenance_recommendations()
        
        with tab6:
            self.render_data_insights()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>🏭 Predictive Maintenance Dashboard | Developed by Keiko Rafi Ananda Prakoso</p>
            <p>Energy Sector AI Solutions | Real-time Monitoring & Predictive Analytics</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    try:
        dashboard = PredictiveMaintenanceDashboard()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"❌ Error loading dashboard: {str(e)}")
        st.info("Please ensure all data files are generated by running the data pipeline first.")

if __name__ == "__main__":
    main() 