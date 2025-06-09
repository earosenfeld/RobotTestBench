"""
Motor simulation module with enhanced features.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MotorParameters:
    """Motor parameters with enhanced features."""
    inertia: float  # kg⋅m²
    damping: float  # N⋅m⋅s/rad
    torque_constant: float  # N⋅m/A
    max_torque: float  # N⋅m
    max_speed: float  # rad/s
    resistance: float  # Ω
    inductance: float  # H
    thermal_mass: float = 0.1  # kg⋅K/J
    thermal_resistance: float = 0.5  # K/W
    gear_ratio: float = 1.0  # -
    gear_efficiency: float = 1.0  # -
    ambient_temp: float = 25.0  # °C
    max_temp: float = 100.0  # °C
    temp_coeff: float = 0.0039  # 1/°C (copper)

@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

class MotorSimulator:
    """Motor simulator with enhanced features."""
    
    def __init__(self, params: MotorParameters):
        """Initialize motor simulator."""
        self.params = params
        self.position = 0.0  # rad
        self.velocity = 0.0  # rad/s
        self.current = 0.0  # A
        self.temperature = params.ambient_temp  # °C
        self._last_update_time = 0.0  # s
        
    def step(self, dt: float, voltage: float) -> tuple[float, float, float]:
        """Step motor simulation forward in time.
        
        Args:
            dt: Time step in seconds
            voltage: Applied voltage in volts
            
        Returns:
            Tuple of (position, velocity, current)
        """
        # Limit voltage to prevent numerical instabilities
        voltage = np.clip(voltage, -24.0, 24.0)
        
        # Calculate electrical dynamics
        voltage_drop = voltage - self.params.torque_constant * self.velocity
        di_dt = (voltage_drop - self.params.resistance * self.current) / self.params.inductance
        
        # Update current with numerical stability
        self.current = np.clip(
            self.current + di_dt * dt,
            -self.params.max_torque / self.params.torque_constant,
            self.params.max_torque / self.params.torque_constant
        )
        
        # Calculate torque with gearbox effects
        motor_torque = self.current * self.params.torque_constant
        output_torque = motor_torque * self.params.gear_ratio * self.params.gear_efficiency
        
        # Calculate mechanical dynamics
        domega_dt = (output_torque - self.params.damping * self.velocity) / self.params.inertia
        self.velocity = np.clip(
            self.velocity + domega_dt * dt,
            -self.params.max_speed,
            self.params.max_speed
        )
        self.position += self.velocity * dt
        
        # Update temperature
        copper_loss = self.current**2 * self.params.resistance
        mechanical_loss = abs(output_torque * self.velocity) * (1.0 - self.params.gear_efficiency)
        total_loss = copper_loss + mechanical_loss
        
        dtemp_dt = (total_loss - (self.temperature - self.params.ambient_temp) / self.params.thermal_resistance) / self.params.thermal_mass
        self.temperature = np.clip(
            self.temperature + dtemp_dt * dt,
            self.params.ambient_temp,
            self.params.max_temp
        )
        
        self._last_update_time += dt
        return self.position, self.velocity, self.current
    
    def get_state(self) -> dict:
        """Get current motor state."""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'current': self.current,
            'torque': self.current * self.params.torque_constant * self.params.gear_ratio * self.params.gear_efficiency,
            'temperature': self.temperature,
            'time': self._last_update_time
        }
    
    def get_temperature(self) -> float:
        """Get current motor temperature."""
        return self.temperature
    
    def get_efficiency(self) -> float:
        """Calculate motor efficiency."""
        input_power = abs(self.current * self.params.torque_constant * self.velocity)
        if input_power < 1e-6:  # Avoid division by zero
            return 1.0
        output_power = abs(self.get_state()['torque'] * self.velocity)
        return output_power / input_power
    
    def get_power_loss(self) -> float:
        """Calculate total power loss."""
        copper_loss = self.current**2 * self.params.resistance
        mechanical_loss = abs(self.get_state()['torque'] * self.velocity) * (1.0 - self.params.gear_efficiency)
        return copper_loss + mechanical_loss

class MotorController:
    """PID controller for motor control."""
    
    def __init__(self, gains: PIDGains, control_mode: str = 'position', output_limits: tuple = (-48.0, 48.0)):
        """Initialize controller with PID gains."""
        self.kp = gains.kp
        self.ki = gains.ki
        self.kd = gains.kd
        self.control_mode = control_mode
        self.output_limits = output_limits
        self.setpoint = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.error = 0.0
        self.max_accel = 50.0  # rad/s² - Added acceleration limit
        
    def set_setpoint(self, setpoint: float):
        """Set the target setpoint."""
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0
        
    def compute(self, measurement: float, dt: float) -> float:
        """Compute control output using PID algorithm with anti-windup."""
        # Calculate error based on control mode
        if self.control_mode == 'position':
            self.error = self.setpoint - measurement
        else:  # velocity control
            self.error = self.setpoint - measurement
            
        # Calculate PID terms
        p_term = self.kp * self.error
        d_term = self.kd * (self.error - self.prev_error) / dt if dt > 0 else 0.0
        
        # Calculate raw output before anti-windup
        output = p_term + self.ki * self.integral + d_term
        
        # Clip output to limits
        output_clipped = np.clip(output, *self.output_limits)
        
        # Anti-windup: only integrate if output is not saturated
        if output == output_clipped:
            self.integral += self.error * dt
        else:
            # Optional: Back-calculation anti-windup
            self.integral += (output_clipped - output) / self.ki if self.ki != 0 else 0.0
            
        # Update previous error
        self.prev_error = self.error
        
        return output_clipped
        
    def get_error(self) -> float:
        """Get the current error."""
        return self.error 