"""Solar energy generation simulation service."""
import numpy as np
import pandas as pd
from datetime import datetime, time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SolarService:
    """Service for simulating solar panel output."""
    
    def __init__(self, capacity_kw: float = 5.0):
        """
        Initialize solar service.
        
        Args:
            capacity_kw: Solar panel capacity in kilowatts
        """
        self.capacity_kw = capacity_kw
    
    def simulate_production(self, 
                          weather_data: pd.DataFrame,
                          latitude: float = 37.7749,
                          longitude: float = -122.4194) -> pd.Series:
        """
        Simulate solar production based on weather data.
        
        Args:
            weather_data: DataFrame with timestamp index and weather conditions
            latitude: Location latitude
            longitude: Location longitude
            
        Returns:
            Series of simulated solar production in kW
        """
        if weather_data.empty:
            return pd.Series(dtype=float)
            
        # Calculate solar position factor (simplified)
        production = pd.Series(index=weather_data.index, dtype=float)
        
        for ts, row in weather_data.iterrows():
            hour = ts.hour + ts.minute/60
            
            # Basic day/night cycle (simplified)
            if 6 <= hour <= 18:  # Daytime
                # Solar position factor (simplified bell curve)
                solar_pos = np.sin(np.pi * (hour - 6) / 12)
                
                # Weather impact
                cloud_factor = 1 - (row.get('cloud_cover', 0) / 200)  # 0-100% -> 0.5-1.0
                rain_factor = 0.3 if row.get('rain', 0) > 0 else 1.0
                
                # Calculate production
                production[ts] = (
                    self.capacity_kw * 
                    max(0, solar_pos) *  # Ensure non-negative
                    cloud_factor * 
                    rain_factor
                )
            else:
                production[ts] = 0.0
                
        return production
    
    def simulate_historical(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate historical solar production data with realistic patterns.
        
        Args:
            dates: DatetimeIndex for the historical period
            
        Returns:
            pd.Series with historical solar production in kW
        """
        # Create a DataFrame with the date range
        df = pd.DataFrame(index=dates)
        
        # Add hour of day (0-23)
        df['hour'] = df.index.hour
        
        # Add day of year (1-365)
        df['day_of_year'] = df.index.dayofyear
        
        # Base production curve (bell curve centered at solar noon)
        # Scale by season (higher in summer, lower in winter)
        df['season_factor'] = 0.7 + 0.3 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)
        
        # Generate base production (bell curve for each day)
        df['production'] = np.exp(-((df['hour'] - 12) ** 2) / 18)  # Bell curve centered at noon
        
        # Apply seasonal variation
        df['production'] *= df['season_factor']
        
        # Add some random noise (10% of max production)
        noise = np.random.normal(0, 0.03, len(df))
        df['production'] = np.clip(df['production'] + noise, 0, None)
        
        # Scale to system capacity
        df['production'] *= self.capacity_kw
        
        # Set production to 0 during night hours
        night_mask = (df['hour'] < 6) | (df['hour'] > 18)
        df.loc[night_mask, 'production'] = 0
        
        # Add some cloudy days (randomly reduce production on some days)
        for day in pd.unique(df.index.date):
            if np.random.random() < 0.2:  # 20% chance of a cloudy day
                cloud_factor = np.random.uniform(0.2, 0.7)  # Reduce to 20-70% of normal
                day_mask = (df.index.date == day)
                df.loc[day_mask, 'production'] *= cloud_factor
        
        return df['production']

    def get_historical_production(self, start: datetime, end: datetime) -> pd.Series:
        """
        Generate mock historical production data.
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            Series of historical production data
        """
        if start >= end:
            return pd.Series()
            
        # Generate timestamps at 15-minute intervals
        timestamps = pd.date_range(start, end, freq='15T')
        
        # Generate mock data with daily pattern and some noise
        hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        
        # Basic daily pattern with noise
        daily_pattern = np.sin((hours - 6) * np.pi / 12)  # 6am to 6pm
        daily_pattern = np.maximum(0, daily_pattern)  # No negative production
        
        # Add some random noise (10% of max capacity)
        noise = np.random.normal(0, 0.1 * self.capacity_kw, size=len(timestamps))
        
        # Combine and ensure non-negative
        production = daily_pattern * self.capacity_kw + noise
        production = np.maximum(0, production)
        
        return pd.Series(production, index=timestamps)
