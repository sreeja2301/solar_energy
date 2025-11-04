"""Weather service for fetching and processing weather data."""
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

from config import OPENWEATHER_API_KEY, DEFAULT_LATITUDE, DEFAULT_LONGITUDE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherService:
    """Service for handling weather data operations."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.use_mock = not (self.api_key and self.api_key != 'your_openweather_api_key_here')
        if self.use_mock:
            logger.warning("Using mock weather data")
    
    def get_forecast(self, lat: float = None, lon: float = None) -> pd.DataFrame:
        """Fetch weather forecast data."""
        if self.use_mock:
            return self._get_mock_forecast()
            
        params = {
            'lat': lat or DEFAULT_LATITUDE,
            'lon': lon or DEFAULT_LONGITUDE,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': 24
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return self._process_forecast_data(response.json())
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return self._get_mock_forecast()
    
    def _process_forecast_data(self, data: dict) -> pd.DataFrame:
        """Process raw forecast data into DataFrame."""
        forecast_items = []
        for item in data.get('list', []):
            forecast_items.append({
                'timestamp': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'cloud_cover': item['clouds']['all'],
                'weather_condition': item['weather'][0]['main'],
                'rain': item.get('rain', {}).get('3h', 0)
            })
        return pd.DataFrame(forecast_items).set_index('timestamp')
    
    def _get_mock_forecast(self) -> pd.DataFrame:
        """Generate mock forecast data."""
        now = datetime.now()
        hours = pd.date_range(now, periods=24, freq='1H')
        return pd.DataFrame({
            'temperature': [20 + 5 * (h.hour/12 - 1) for h in hours],
            'cloud_cover': [30 + 40 * (h.hour % 24)/24 for h in hours],
            'weather_condition': ['Clear' if h.hour > 6 and h.hour < 20 else 'Clouds' for h in hours],
            'rain': [0.0] * 24
        }, index=hours)
