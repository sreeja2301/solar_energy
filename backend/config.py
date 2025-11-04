"""
Configuration settings for the Predictive Solar Energy Optimizer.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'your_openweather_api_key_here')

# Application Settings
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Simulation Parameters
DEFAULT_LATITUDE = 37.7749  # Default location: San Francisco
DEFAULT_LONGITUDE = -122.4194
SIMULATION_INTERVAL_MINUTES = 15  # Time between simulation steps

# Battery Parameters
BATTERY_CAPACITY_KWH = 10.0  # 10 kWh battery
BATTERY_EFFICIENCY = 0.95  # 95% round-trip efficiency
MIN_BATTERY_LEVEL = 0.1  # Minimum charge level (10%)
MAX_BATTERY_LEVEL = 1.0  # Maximum charge level (100%)

# Solar Panel Parameters
SOLAR_PANEL_CAPACITY_KW = 5.0  # 5 kW solar panel system

# File Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
