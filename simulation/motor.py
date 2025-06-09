"""
Motor simulation module for RobotTestBench.
Implements a basic motor model with inertia, friction, and load parameters.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MotorParameters:
    """Motor physical parameters."""
    inertia: float = 0.05  # kg⋅m²
    damping: float = 0.5  # N⋅m⋅s/rad
    torque_constant: float = 0.2  # N⋅m/A
    max_torque: float = 2.0  # N⋅m
    max_speed: float = 20.0  # rad/s
    resistance: float = 1.0  # Ω
    inductance: float = 0.1  # H

@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

class MotorSimulator:
    """Simulates a DC motor with electrical and mechanical dynamics."""
    
    def __init__(self, params: MotorParameters):
        """Initialize motor with parameters."""
        self.params = params
        self.position = 0.0
        self.velocity = 0.0
        self.current = 0.0
        self.torque = 0.0
        self.max_accel = 50.0  # rad/s² - Added acceleration limit
        
    def get_state(self) -> dict:
        """Get current motor state."""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'current': self.current,
            'torque': self.torque
        }
        
    def step(self, voltage: float, load_torque: float, dt: float):
        """Step motor simulation forward in time."""
        # Calculate current
        di_dt = (voltage - self.params.resistance * self.current - 
                self.params.torque_constant * self.velocity) / self.params.inductance
        self.current += di_dt * dt
        
        # Calculate torque
        self.torque = self.params.torque_constant * self.current
        
        # Limit torque
        self.torque = np.clip(self.torque, -self.params.max_torque, self.params.max_torque)
        
        # Calculate acceleration
        acceleration = (self.torque - self.params.damping * self.velocity - load_torque) / self.params.inertia
        
        # Limit acceleration
        acceleration = np.clip(acceleration, -self.max_accel, self.max_accel)
        
        # Update velocity and position
        self.velocity += acceleration * dt
        self.velocity = np.clip(self.velocity, -self.params.max_speed, self.params.max_speed)
        self.position += self.velocity * dt
        
        # Print debug info
        print(f"MotorSim Step: pos={self.position:.4f}, vel={self.velocity:.4f}, accel={acceleration:.4f}, dt={dt:.4f}")
        return self.position, self.velocity, self.current
        
    def reset(self):
        """Reset motor state to initial conditions."""
        self.position = 0.0
        self.velocity = 0.0
        self.current = 0.0
        self.torque = 0.0 

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