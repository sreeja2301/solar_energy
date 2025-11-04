"""
Predictive Solar Energy Optimizer - Main Application

This module initializes and runs the solar energy optimization system.
"""
import logging
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
from dotenv import load_dotenv

from services.weather import WeatherService
from services.solar import SolarService
from services.battery import Battery
from services.optimizer import EnergyOptimizer, generate_demand_profile
from services.ml_models import EnergyAIAssistant, generate_energy_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SolarEnergyOptimizer:
    """Main application class for the solar energy optimizer."""
    
    def __init__(self):
        """Initialize the solar energy optimizer with services."""
        # Initialize services
        self.weather_service = WeatherService(api_key=os.getenv('OPENWEATHER_API_KEY'))
        self.solar_service = SolarService(
            capacity_kw=float(os.getenv('SOLAR_CAPACITY_KW', 5.0))
        )
        self.battery = Battery(
            capacity_kwh=float(os.getenv('BATTERY_CAPACITY_KWH', 10.0)),
            max_charge_rate_kw=float(os.getenv('BATTERY_MAX_CHARGE_RATE_KW', 5.0)),
            efficiency=float(os.getenv('BATTERY_EFFICIENCY', 0.95))
        )
        self.optimizer = EnergyOptimizer({
            'battery_capacity_kwh': float(os.getenv('BATTERY_CAPACITY_KWH', 10.0)),
            'max_charge_rate_kw': float(os.getenv('BATTERY_MAX_CHARGE_RATE_KW', 5.0))
        })
        self.ai_assistant = EnergyAIAssistant()
        self.optimization_results = []
        
        # Load or generate historical data for ML training
        self.historical_data = self._generate_historical_data()
        self.optimizer.train_models(self.historical_data)
    
    def _generate_historical_data(self, days: int = 90) -> pd.DataFrame:
        """Generate or load historical data for ML training."""
        # Create date range with hourly frequency
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate load profile for the exact date range
        load = generate_demand_profile(hours=len(dates), resolution_min=60)  # 60-minute resolution for hourly data
        
        # Simulate solar production
        solar = self.solar_service.simulate_historical(dates=dates)
        
        # Create DataFrame with consistent lengths
        data = {
            'timestamp': dates,
            'load_kw': load.values[:len(dates)],  # Ensure we take only the required number of values
            'solar_kw': solar.values if hasattr(solar, 'values') else solar[:len(dates)],
            'battery_soc': np.random.uniform(0.2, 0.8, len(dates))
        }
        
        # Create DataFrame and set index
        df = pd.DataFrame(data)
        return df.set_index('timestamp')
    
    def run_forecast(self, days: int = 3) -> pd.DataFrame:
        """Run the energy optimization forecast."""
        try:
            # Get weather forecast
            forecast = self.weather_service.get_forecast()
            
            # Simulate solar production
            solar_forecast = self.solar_service.simulate_production(
                weather_data=forecast
            )
            
            # Generate load profile
            load_forecast = generate_demand_profile(hours=len(forecast))
            
            # Run optimization for each time step
            results = []
            for i, (ts, row) in enumerate(forecast.iterrows()):
                current_state = {
                    'battery_soc': self.battery.soc,
                    'solar_kw': solar_forecast.iloc[i],
                    'load_kw': load_forecast.iloc[i]
                }
                
                # Get optimization result
                result = self.optimizer.optimize_energy_flow(
                    current_state=current_state,
                    forecast=pd.DataFrame({
                        'solar_kw': solar_forecast.iloc[i:i+24],
                        'load_kw': load_forecast.iloc[i:i+24]
                    })
                )
                
                # Update battery state
                self.battery.step(
                    result['battery_power_kw'],
                    duration_hours=1.0
                )
                
                # Store results
                result.update({
                    'timestamp': ts,
                    'solar_kw': solar_forecast.iloc[i],
                    'load_kw': load_forecast.iloc[i],
                    'battery_soc': self.battery.soc,
                    'net_export_kw': solar_forecast.iloc[i] - load_forecast[i] - result['battery_power_kw']
                })
                results.append(result)
            
            return pd.DataFrame(results).set_index('timestamp')
            
        except Exception as e:
            logger.error(f"Error running forecast: {e}")
            raise

def display_battery_recommendations(df):
    """Display battery optimization recommendations."""
    st.subheader("🔋 Smart Battery Assistant")
    
    if df.empty:
        st.warning("No forecast data available.")
        return
        
    # Get current state
    current_data = df.iloc[0]
    
    # Generate AI insights
    ai_insights = st.session_state.optimizer.ai_assistant.generate_insights({
        'solar_kw': current_data.get('solar_kw', 0),
        'load_kw': current_data.get('load_kw', 0),
        'battery_soc': current_data.get('battery_soc', 0),
        'net_export_kw': current_data.get('net_export_kw', 0)
    })
    
    # Display AI insights in a nice info box
    with st.expander("🤖 AI Energy Assistant", expanded=True):
        st.info(ai_insights)
        
        # Add quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💡 Energy Saving Tips"):
                st.session_state.ai_message = "🔌 Try running high-power appliances during peak solar generation (10 AM - 2 PM) to maximize self-consumption."
        with col2:
            if st.button("🔋 Battery Advice"):
                st.session_state.ai_message = "🔋 For optimal battery health, try to keep charge between 20-80% most of the time."
        with col3:
            if st.button("🌦️ Weather Impact"):
                st.session_state.ai_message = "🌦️ Cloudy weather expected tomorrow - consider pre-charging your battery tonight on off-peak rates."
        
        # Display any button-triggered messages
        if hasattr(st.session_state, 'ai_message'):
            st.success(st.session_state.ai_message)
    
    # Display battery status
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Battery SOC", f"{current_data.get('battery_soc', 0)*100:.1f}%")
        
        # Battery status visualization
        soc = current_data.get('battery_soc', 0)
        color = "green" if soc > 0.5 else "orange" if soc > 0.2 else "red"
        st.progress(soc, text=f"Battery Level: {soc*100:.1f}%")
        
    with col2:
        st.metric("Current Solar Production", f"{current_data.get('solar_kw', 0):.2f} kW")
        st.metric("Current Load", f"{current_data.get('load_kw', 0):.2f} kW")
    
    # Show ML confidence if available
    if 'ml_confidence' in current_data:
        confidence = current_data.get('ml_confidence', 0)
        if confidence > 0:
            st.info(f"🤖 ML Model Confidence: {confidence*100:.0f}%")
    
    # Show forecast charts
    st.subheader("📈 Energy Forecast")
    
    # Prepare forecast data
    forecast_hours = min(24, len(df))
    forecast_df = pd.DataFrame({
        'Timestamp': df.index[:forecast_hours],
        'Solar (kW)': df['solar_kw'].iloc[:forecast_hours],
        'Load (kW)': df['load_kw'].iloc[:forecast_hours],
        'Battery SOC (%)': df['battery_soc'].iloc[:forecast_hours] * 100,
        'Net Export (kW)': df.get('net_export_kw', 0).iloc[:forecast_hours]
    }).set_index('Timestamp')
    
    # Plot energy forecast
    st.line_chart(forecast_df[['Solar (kW)', 'Load (kW)', 'Net Export (kW)']])
    
    # Plot battery SOC separately for better scaling
    st.line_chart(forecast_df[['Battery SOC (%)']])
    
    # Add a section for AI-generated report
    if st.button("📊 Generate Detailed Energy Report"):
        report = generate_energy_report(df)
        st.markdown(report)

def main():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="🤖 AI-Powered Solar Energy Optimizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌞 AI-Powered Solar Energy Optimizer")
    st.markdown("""
    Welcome to your smart solar energy management system! This AI-powered tool helps you optimize 
    your solar energy usage, reduce grid dependency, and save money.
    """)
    
    # Initialize session state
    if 'optimizer' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.optimizer = SolarEnergyOptimizer()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        days = st.slider("Forecast Days", 1, 7, 3)
        
        if st.button("🔄 Run Forecast"):
            with st.spinner("Running AI-powered forecast..."):
                try:
                    results = st.session_state.optimizer.run_forecast(days=days)
                    st.session_state.results = results
                    st.success("AI forecast completed successfully!")
                except Exception as e:
                    st.error(f"Error running forecast: {e}")
        
        st.markdown("---")
        st.markdown("### System Status")
        if 'results' in st.session_state:
            latest = st.session_state.results.iloc[0]
            st.metric("Battery Level", f"{latest['battery_soc']*100:.1f}%")
            st.metric("Current Solar", f"{latest['solar_kw']:.2f} kW")
            st.metric("Current Load", f"{latest['load_kw']:.2f} kW")
    
    # Main content area
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Dashboard metrics
        st.subheader("📊 Real-time Dashboard")
        
        # Top row: Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Solar Generation", f"{results['solar_kw'].sum():.1f} kWh")
        with col2:
            st.metric("Total Consumption", f"{results['load_kw'].sum():.1f} kWh")
        with col3:
            st.metric("Battery SOC", f"{results['battery_soc'].iloc[-1]*100:.1f}%")
        with col4:
            net_export = results['solar_kw'].sum() - results['load_kw'].sum()
            st.metric("Net Export", f"{net_export:.1f} kWh")
        
        # AI Recommendations
        display_battery_recommendations(results)
        
        # Energy flow chart
        st.subheader("⚡ Energy Flow")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=results.index, y=results['solar_kw'], name="Solar Generation", stackgroup='one'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=results.index, y=results['load_kw'], name="Load", line=dict(dash='dash')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=results.index, y=results['battery_soc'] * 10, name="Battery SOC (scaled)", 
                      line=dict(color='purple', width=2)),
            secondary_y=True,
        )
        
        # Add figure layout
        fig.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            hovermode="x unified"
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
        fig.update_yaxes(title_text="Battery SOC (%)", secondary_y=True, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI-Generated Report
        if st.button("📊 Generate AI Energy Report"):
            with st.spinner("Generating AI-powered energy insights..."):
                report = st.session_state.optimizer.optimizer.generate_report(results)
                st.markdown(report)
        
        # Raw data view
        if st.checkbox("📋 Show Raw Data"):
            st.dataframe(results)
    
    else:
        # Show welcome/instructions if no forecast has been run
        st.info("""
        ### Getting Started
        1. Configure your forecast settings in the sidebar
        2. Click "Run Forecast" to start the AI-powered optimization
        3. View your energy insights and recommendations
        
        The system uses advanced machine learning to:
        - Predict your energy usage patterns
        - Optimize battery charging/discharging
        - Provide personalized energy-saving recommendations
        """)

if __name__ == "__main__":
    main()
