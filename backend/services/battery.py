"""Battery model for energy storage simulation."""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Battery:
    """Simple battery model with charge/discharge behavior."""
    
    def __init__(self, 
                capacity_kwh: float = 10.0, 
                max_charge_rate_kw: float = 5.0,
                max_discharge_rate_kw: float = 5.0,
                efficiency: float = 0.95,
                initial_soc: float = 0.5):
        """
        Initialize battery model.
        
        Args:
            capacity_kwh: Total energy capacity in kWh
            max_charge_rate_kw: Maximum charging power in kW
            max_discharge_rate_kw: Maximum discharging power in kW
            efficiency: Round-trip efficiency (0-1)
            initial_soc: Initial state of charge (0-1)
        """
        self.capacity_kwh = capacity_kwh
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw
        self.efficiency = efficiency
        self._soc = max(0.0, min(1.0, initial_soc))  # Clamp between 0-1
        
        # Track metrics
        self.energy_stored_kwh = self._soc * self.capacity_kwh
        self.total_charged_kwh = 0.0
        self.total_discharged_kwh = 0.0
    
    @property
    def soc(self) -> float:
        """
        Get the current state of charge as a value between 0 and 1.
        
        Returns:
            float: State of charge (0.0 to 1.0)
        """
        return self.energy_stored_kwh / self.capacity_kwh if self.capacity_kwh > 0 else 0.0

    def step(self, power_kw: float, duration_hours: float = 1.0) -> float:
        """
        Simulate battery operation for one time step.
        
        Args:
            power_kw: Power flow in kW (positive for charging, negative for discharging)
            duration_hours: Duration of the time step in hours
            
        Returns:
            Actual power flow after considering battery constraints (kW)
        """
        if duration_hours <= 0:
            return 0.0
            
        # Calculate energy flow
        energy_kwh = power_kw * duration_hours
        
        # Apply efficiency (charging loses energy)
        if energy_kwh > 0:  # Charging
            energy_kwh *= np.sqrt(self.efficiency)  # Single-trip efficiency
        elif energy_kwh < 0:  # Discharging
            energy_kwh /= np.sqrt(self.efficiency)  # Single-trip efficiency
        
        # Calculate new state of charge
        new_energy = self.energy_stored_kwh + energy_kwh
        
        # Apply capacity constraints
        if new_energy > self.capacity_kwh:
            energy_kwh = self.capacity_kwh - self.energy_stored_kwh
            new_energy = self.capacity_kwh
            actual_power = energy_kwh / duration_hours
        elif new_energy < 0:
            energy_kwh = -self.energy_stored_kwh
            new_energy = 0.0
            actual_power = energy_kwh / duration_hours
        else:
            actual_power = power_kw
        
        # Apply charge/discharge rate limits
        if actual_power > self.max_charge_rate_kw:
            actual_power = self.max_charge_rate_kw
            energy_kwh = actual_power * duration_hours
            new_energy = self.energy_stored_kwh + energy_kwh
        elif actual_power < -self.max_discharge_rate_kw:
            actual_power = -self.max_discharge_rate_kw
            energy_kwh = actual_power * duration_hours
            new_energy = self.energy_stored_kwh + energy_kwh
        
        # Update state
        self.energy_stored_kwh = max(0.0, min(self.capacity_kwh, new_energy))
        
        # Update metrics
        if energy_kwh > 0:
            self.total_charged_kwh += energy_kwh
        else:
            self.total_discharged_kwh += abs(energy_kwh)
        
        return actual_power
    
    def get_state(self) -> dict:
        """Get current battery state."""
        return {
            'soc': self.soc,
            'energy_stored_kwh': self.energy_stored_kwh,
            'energy_remaining_kwh': self.energy_stored_kwh,
            'energy_capacity_kwh': self.capacity_kwh,
            'max_charge_rate_kw': self.max_charge_rate_kw,
            'max_discharge_rate_kw': self.max_discharge_rate_kw,
            'efficiency': self.efficiency,
            'total_charged_kwh': self.total_charged_kwh,
            'total_discharged_kwh': self.total_discharged_kwh
        }
    
    def reset(self, initial_soc: Optional[float] = None):
        """Reset battery to initial state."""
        if initial_soc is not None:
            self._soc = max(0.0, min(1.0, initial_soc))
        self.energy_stored_kwh = self._soc * self.capacity_kwh
        self.total_charged_kwh = 0.0
        self.total_discharged_kwh = 0.0
