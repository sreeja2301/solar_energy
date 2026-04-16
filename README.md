# Predictive Solar Energy Optimizer

A smart energy management system that optimizes solar energy usage, battery storage, and grid interaction using weather forecasts and machine learning.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

## Overview
The Predictive Solar Energy Optimizer aims to reduce grid reliance and maximize the efficiency of residential or commercial solar setups. By combining real-time weather forecasting through OpenWeatherMap with advanced predictive models, this system anticipates both solar generation and power consumption patterns, intelligently routing energy to battery storage or consumption as needed.

## Key Features
- **Weather Integration**: Fetches real-time weather data from the OpenWeatherMap API for accurate yield predictions.
- **Solar Production**: Simulates photovoltaic (PV) array outputs in response to hourly and seasonal weather conditions.
- **Demand Prediction**: Maps daily human and appliance usage patterns to forecast future consumption.
- **Battery Management**: Provides an algorithmic charge and discharge strategy to maximize battery lifespan and minimize peak grid loads.
- **Energy Optimization**: Automatically calculates the optimal energy flow between Solar, Grid, Battery, and Home Load.
- **Interactive Dashboard**: Real-time visualization via Streamlit for deep insights into energy parameters.
- **Machine Learning**: Dedicated predictive models refine historical data to enhance forecasting capabilities over time.

## Architecture
The application runs as a Python Streamlit app backed by several domain-specific microservices:
1. **Data Ingestion**: Connects to OpenWeatherMap for incoming weather states.
2. **Machine Learning Layer**: Processes weather vectors to predict PV production (`ml_models.py`).
3. **Optimization Engine**: Analyzes cost per kWh, current battery State of Charge (SoC), and forecasted load to determine optimal power routing (`optimizer.py`).
4. **UI Layer**: A Streamlit-based web dashboard (`app.py`) for user interaction and visualization.

## Getting Started

### Prerequisites
- Python 3.8+
- OpenWeatherMap API key (free tier available at OpenWeatherMap website)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/predictive-solar-optimizer.git
   cd predictive-solar-optimizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API key:
   ```
   OPENWEATHER_API_KEY=your_api_key_here
   ```

### Running the Application

Start the Streamlit dashboard:
```bash
streamlit run backend/app.py
```

Open your browser and navigate to `http://localhost:8501`

## Project Structure

```text
predictive-solar-optimizer/
|-- backend/               
|   |-- services/         
|   |   |-- weather.py    
|   |   |-- solar.py      
|   |   |-- battery.py    
|   |   |-- ml_models.py  
|   |   |-- optimizer.py  
|   |-- config.py         
|   |-- app.py            
|-- data/                 
|   |-- raw/             
|   |-- processed/       
|-- requirements.txt     
|-- README.md           
```

## Configuration
Edit `backend/config.py` to customize hardware and environmental constraints:
- Solar panel capacity (kWp)
- Battery specifications (Capacity, Charge Rate, Discharge Rate)
- Default location (Latitude/Longitude)
- Simulation time-steps

## Future Enhancements
- [ ] Add support for real hardware integration (Raspberry Pi / Inverter APIs)
- [ ] Implement enhanced deep learning for granular demand prediction
- [ ] Add support for multiple distributed energy sources (e.g., wind turbines)
- [ ] Implement user authentication and multi-tenant support for utility companies
- [ ] Add mobile app notification integrations

---

## Author

- Name: Sreeja Bonthu
- mail: sreejabonthu@gmail.com
