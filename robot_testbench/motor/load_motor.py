"""
Load motor simulation module with mechanical coupling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .motor_simulator import MotorParameters, MotorSimulator

@dataclass
class LoadMotorParameters:
    """Parameters for load motor simulation."""
    inertia: float  # kg⋅m²
    damping: float  # N⋅m⋅s/rad
    friction: float  # N⋅m
    back_emf_constant: float  # V⋅s/rad
    max_speed: float  # rad/s
    coupling_stiffness: float = 1000.0  # N⋅m/rad
    coupling_damping: float = 10.0  # N⋅m⋅s/rad
    coupling_backlash: float = 0.001  # rad

class LoadMotor:
    """Load motor simulator with mechanical coupling."""
    
    def __init__(self, params: LoadMotorParameters):
        """Initialize load motor simulator."""
        self.params = params
        self.position = 0.0  # rad
        self.velocity = 0.0  # rad/s
        self.acceleration = 0.0  # rad/s²
        self.torque = 0.0  # N⋅m
        self.back_emf = 0.0  # V
        self._last_update_time = 0.0  # s
        
    def step(self, dt: float, drive_motor_position: float, drive_motor_velocity: float) -> Tuple[float, float, float]:
        """Step load motor simulation forward in time.
        
        Args:
            dt: Time step in seconds
            drive_motor_position: Position of the drive motor in radians
            drive_motor_velocity: Velocity of the drive motor in rad/s
            
        Returns:
            Tuple of (position, velocity, torque)
        """
        # Calculate coupling torque
        position_error = drive_motor_position - self.position
        velocity_error = drive_motor_velocity - self.velocity
        
        # Apply backlash
        if abs(position_error) < self.params.coupling_backlash:
            position_error = 0.0
        
        # Calculate coupling torque
        coupling_torque = (
            self.params.coupling_stiffness * position_error +
            self.params.coupling_damping * velocity_error
        )
        
        # Calculate back-EMF
        self.back_emf = self.params.back_emf_constant * self.velocity
        
        # Calculate total torque
        friction_torque = np.sign(self.velocity) * self.params.friction
        damping_torque = self.params.damping * self.velocity
        
        self.torque = coupling_torque - friction_torque - damping_torque
        
        # Calculate acceleration
        self.acceleration = self.torque / self.params.inertia
        
        # Update velocity with numerical stability
        self.velocity = np.clip(
            self.velocity + self.acceleration * dt,
            -self.params.max_speed,
            self.params.max_speed
        )
        
        # Update position
        self.position += self.velocity * dt
        
        self._last_update_time += dt
        return self.position, self.velocity, self.torque
    
    def get_state(self) -> dict:
        """Get current load motor state."""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'torque': self.torque,
            'back_emf': self.back_emf,
            'time': self._last_update_time
        }

class Load:
    """Configurable test load simulator."""
    
    def __init__(self):
        """Initialize test load simulator."""
        self.load_type = 'constant_torque'  # constant_torque, inertial, friction_ramp, regenerative
        self.constant_torque = 0.0  # N⋅m
        self.inertia = 0.0  # kg⋅m²
        self.friction_start = 0.0  # N⋅m
        self.friction_end = 0.0  # N⋅m
        self.friction_ramp_time = 0.0  # s
        self.regenerative_voltage = 0.0  # V
        self._start_time = 0.0  # s
        
    def set_load_type(self, load_type: str, **kwargs):
        """Set the type of load and its parameters."""
        self.load_type = load_type
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._start_time = 0.0
    
    def get_load_torque(self, velocity: float, dt: float) -> float:
        """Calculate load torque based on current configuration."""
        self._start_time += dt
        
        if self.load_type == 'constant_torque':
            return self.constant_torque
            
        elif self.load_type == 'inertial':
            return self.inertia * velocity
            
        elif self.load_type == 'friction_ramp':
            ramp_progress = min(1.0, self._start_time / self.friction_ramp_time)
            friction = self.friction_start + (self.friction_end - self.friction_start) * ramp_progress
            return np.sign(velocity) * friction
            
        elif self.load_type == 'regenerative':
            # Simulate regenerative braking effect
            back_emf = velocity * self.regenerative_voltage
            return np.sign(velocity) * back_emf  # Positive torque opposes motion
            
        return 0.0 