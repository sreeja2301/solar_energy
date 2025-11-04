"""Energy optimization with ML capabilities."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from .ml_models import LSTMForecaster, BatteryEnv, train_rl_agent, EnergyAIAssistant, generate_energy_report

logger = logging.getLogger(__name__)

class EnergyOptimizer:
    """Optimize energy usage with ML assistance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.forecaster = LSTMForecaster()
        self.ai_assistant = EnergyAIAssistant()
        self.rl_agent = None
        self.is_trained = False
        
    def train_models(self, historical_data: pd.DataFrame) -> None:
        """Train ML models on historical data."""
        try:
            # Train load forecaster
            self.forecaster.train(historical_data['load_kw'])
            
            # Train RL agent for battery optimization
            env = BatteryEnv(
                solar_gen=historical_data['solar_kw'].values,
                load=historical_data['load_kw'].values,
                battery_capacity=self.config.get('battery_capacity_kwh', 10.0),
                max_charge_rate=self.config.get('max_charge_rate_kw', 5.0)
            )
            self.rl_agent = train_rl_agent(env)
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            self.is_trained = False
    
    def optimize_energy_flow(self, current_state: Dict[str, Any], 
                           forecast: pd.DataFrame) -> Dict[str, Any]:
        """Optimize energy flow using ML models."""
        try:
            if not self.is_trained:
                return self._fallback_optimization(current_state, forecast)
                
            # Get ML-based load forecast
            load_forecast = self.forecaster.forecast(
                forecast['load_kw'],
                steps=min(24, len(forecast))
            )
            
            # Use RL agent for battery control if available
            if self.rl_agent is not None:
                battery_action = self.rl_agent.predict(
                    np.array([
                        current_state['battery_soc'],
                        forecast['solar_kw'].iloc[0],
                        forecast['load_kw'].iloc[0],
                        pd.Timestamp.now().hour
                    ]),
                    deterministic=True
                )[0]
            else:
                battery_action = 0  # Fallback to no action
            
            # Calculate energy flows
            solar_power = forecast['solar_kw'].iloc[0]
            load_power = forecast['load_kw'].iloc[0]
            
            # Apply battery action
            battery_power = battery_action * self.config.get('max_charge_rate_kw', 5.0)
            
            # Calculate grid interaction
            grid_import = max(0, load_power - solar_power - max(0, battery_power))
            grid_export = max(0, solar_power - load_power - max(0, -battery_power))
            
            # Generate AI insights
            insights = self.ai_assistant.generate_insights({
                'solar_kw': solar_power,
                'load_kw': load_power,
                'battery_soc': current_state['battery_soc'],
                'battery_power': battery_power,
                'grid_import': grid_import,
                'grid_export': grid_export
            })
            
            return {
                'battery_power_kw': battery_power,
                'grid_import_kw': grid_import,
                'grid_export_kw': grid_export,
                'recommendations': insights,
                'load_forecast': load_forecast,
                'ml_confidence': 0.9  # Confidence score for ML predictions
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self._fallback_optimization(current_state, forecast)
    
    def _fallback_optimization(self, current_state: Dict[str, Any], 
                             forecast: pd.DataFrame) -> Dict[str, Any]:
        """Fallback optimization without ML."""
        solar_power = forecast['solar_kw'].iloc[0]
        load_power = forecast['load_kw'].iloc[0]
        
        # Simple rule-based optimization
        if solar_power > load_power:
            # Charge battery with excess solar
            battery_power = min(
                self.config.get('max_charge_rate_kw', 5.0),
                (1 - current_state['battery_soc']) * self.config.get('battery_capacity_kwh', 10.0)
            )
            grid_import = 0
            grid_export = max(0, solar_power - load_power - battery_power)
        else:
            # Discharge battery to cover load
            battery_power = max(
                -self.config.get('max_charge_rate_kw', 5.0),
                -current_state['battery_soc'] * self.config.get('battery_capacity_kwh', 10.0)
            )
            grid_import = max(0, load_power - solar_power + battery_power)
            grid_export = 0
        
        return {
            'battery_power_kw': battery_power,
            'grid_import_kw': grid_import,
            'grid_export_kw': grid_export,
            'recommendations': "Using basic optimization (ML models not available)",
            'load_forecast': forecast['load_kw'].values[:24],
            'ml_confidence': 0.0
        }
    
    def generate_report(self, data: pd.DataFrame) -> str:
        """Generate an AI-powered energy report."""
        return generate_energy_report(data)

def generate_demand_profile(hours: int = 24, resolution_min: int = 15) -> pd.Series:
    """Generate synthetic load profile."""
    steps = int(hours * 60 / resolution_min)
    t = np.linspace(0, 24, steps)
    
    # Base + daily pattern + noise
    base = 0.2  # kW
    daily = 0.8 * np.sin(2 * np.pi * (t - 8) / 24)
    noise = 0.1 * np.random.normal(0, 1, steps)
    
    # Combine and ensure positive
    load = np.maximum(0.1, base + daily + noise)
    
    # Create time index
    index = pd.date_range(
        start=pd.Timestamp.now().normalize(),
        periods=steps,
        freq=f'{resolution_min}T'
    )
    
    return pd.Series(load, index=index, name='load_kw')
