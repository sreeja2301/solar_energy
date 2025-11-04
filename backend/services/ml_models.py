"""
Machine Learning Models for Solar Energy Optimization
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import random
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LSTMForecaster:
    """LSTM model for energy load forecasting."""
    
    def __init__(self, lookback: int = 24, epochs: int = 50, batch_size: int = 32):
        self.lookback = lookback  # hours of historical data to use
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        
    def _build_model(self) -> tf.keras.Model:
        """Build and compile the LSTM model."""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def preprocess_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback):
            X.append(scaled_data[i:(i + self.lookback)])
            y.append(scaled_data[i + self.lookback])
            
        return np.array(X), np.array(y)
    
    def train(self, train_data: pd.Series) -> None:
        """Train the LSTM model."""
        X, y = self.preprocess_data(train_data)
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def forecast(self, history: pd.Series, steps: int = 24) -> np.ndarray:
        """Generate load forecast for future steps."""
        scaled_history = self.scaler.transform(history.values.reshape(-1, 1))
        predictions = []
        
        current_batch = scaled_history[-self.lookback:].reshape(1, self.lookback, 1)
        
        for _ in range(steps):
            current_pred = self.model.predict(current_batch, verbose=0)
            predictions.append(current_pred[0][0])
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred[0]]], axis=1)
            
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


class BatteryEnv(gym.Env):
    """Custom environment for battery optimization using RL."""
    
    def __init__(self, solar_gen: np.ndarray, load: np.ndarray, 
                 battery_capacity: float = 10.0, max_charge_rate: float = 5.0):
        super(BatteryEnv, self).__init__()
        
        self.solar_gen = solar_gen
        self.load = load
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        
        # Action space: -1 (discharge) to 1 (charge), 0 (do nothing)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: [battery_soc, current_solar, current_load, hour_of_day]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, np.inf, np.inf, 23]),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.battery_soc = 0.5  # Start at 50% charge
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time step within the environment."""
        # Get current state
        current_solar = self.solar_gen[self.current_step]
        current_load = self.load[self.current_step]
        hour = self.current_step % 24
        
        # Calculate battery action
        charge_rate = action[0] * self.max_charge_rate
        
        # Update battery state
        energy_available = current_solar - current_load
        
        # Charge battery if excess solar
        if energy_available > 0 and charge_rate > 0:
            charge_energy = min(charge_rate, energy_available, 
                              self.battery_capacity - self.battery_soc)
            self.battery_soc += charge_energy
            energy_available -= charge_energy
        # Discharge battery if needed
        elif energy_available < 0 and charge_rate < 0:
            discharge_energy = min(-charge_rate, -energy_available, self.battery_soc)
            self.battery_soc -= discharge_energy
            energy_available += discharge_energy
            
        # Calculate reward (negative cost)
        grid_import = max(0, -energy_available)
        grid_export = max(0, energy_available)
        
        # Reward function components
        cost = grid_import * 0.15  # Cost of importing from grid ($/kWh)
        revenue = grid_export * 0.10  # Revenue from exporting to grid ($/kWh)
        battery_penalty = -0.01 * (self.battery_soc - 0.5) ** 2  # Prefer ~50% charge
        
        reward = -cost + revenue + battery_penalty
        
        # Update step
        self.current_step += 1
        done = self.current_step >= len(self.solar_gen) - 1
        
        # Next state
        next_state = np.array([
            self.battery_soc / self.battery_capacity,
            self.solar_gen[min(self.current_step, len(self.solar_gen)-1)],
            self.load[min(self.current_step, len(self.load)-1)],
            (hour + 1) % 24
        ])
        
        return next_state, reward, done, {}
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.current_step = 0
        self.battery_soc = 0.5 * self.battery_capacity
        return np.array([self.battery_soc, self.solar_gen[0], self.load[0], 0])


def train_rl_agent(env: gym.Env, episodes: int = 100) -> Any:
    """Train a reinforcement learning agent for battery optimization."""
    try:
        # Using Stable Baselines3 for RL (would be imported if available)
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Wrap environment
        env = DummyVecEnv([lambda: env])
        
        # Initialize PPO agent
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log="./tensorboard_logs/"
        )
        
        # Train the agent
        model.learn(total_timesteps=episodes * len(env.get_attr('solar_gen')[0]))
        
        return model
        
    except ImportError:
        logger.warning("Stable Baselines3 not installed. Using a dummy model.")
        return None


class EnergyAIAssistant:
    """GenAI assistant for energy management."""
    
    def __init__(self):
        self.system_prompt = """You are an AI energy assistant helping users optimize their 
        solar energy usage. Provide clear, actionable advice based on their energy data."""
    
    def generate_insights(self, data: Dict[str, Any]) -> str:
        """Generate natural language insights from energy data."""
        # In a real implementation, this would use an LLM API
        solar = data.get('solar_kw', 0)
        load = data.get('load_kw', 0)
        battery = data.get('battery_soc', 0) * 100
        
        if solar > load * 1.5:
            return ("☀️ Great news! You're generating {solar:.1f}kW while only using {load:.1f}kW. "
                   "Consider charging your battery (currently at {battery:.0f}%) or running "
                   "high-power appliances now.")
        elif solar < load * 0.5:
            return ("🌧️ Limited solar generation ({solar:.1f}kW) compared to your load ({load:.1f}kW). "
                   "Your battery is at {battery:.0f}% - consider reducing non-essential power use.")
        else:
            return ("🌤️ Your system is balanced - generating {solar:.1f}kW with {load:.1f}kW load. "
                   "Battery at {battery:.0f}% - good for now!")


def generate_energy_report(data: pd.DataFrame) -> str:
    """Generate a detailed energy report using AI."""
    # In a real implementation, this would use an LLM API
    total_solar = data['solar_kw'].sum()
    total_load = data['load_kw'].sum()
    max_import = data['grid_import_kw'].max()
    
    return f"""
    # 🌞 Energy Report Summary
    
    ## 📊 Key Metrics
    - Total Solar Generation: {total_solar:.1f} kWh
    - Total Energy Consumption: {total_load:.1f} kWh
    - Peak Grid Import: {max_import:.1f} kW
    
    ## 💡 Recommendations
    1. {'Consider adding more solar panels' if total_solar/total_load < 0.7 else 'Great solar coverage!'}
    2. {'Your peak import is high - consider load shifting' if max_import > 5.0 else 'Good load management'}
    3. Battery utilization: {'excellent' if data['battery_soc'].std() > 0.2 else 'could be improved'}
    
    ## 📈 Next Steps
    - Review your hourly usage patterns
    - Consider time-of-use rates for additional savings
    - Check for firmware updates on your inverter
    """
