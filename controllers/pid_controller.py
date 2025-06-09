"""
PID controller implementation for RobotTestBench.
Supports position, velocity, and torque control modes.
"""

from dataclasses import dataclass
from typing import Literal
from simple_pid import PID

@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

class MotorController:
    """Basic PID-based motor controller with multiple control modes."""
    
    def __init__(self, gains: PIDGains, control_mode: Literal['position', 'velocity', 'torque'] = 'position', output_limits=(-12.0, 12.0)):
        self.control_mode = control_mode
        self.pid = PID(
            Kp=gains.kp,
            Ki=gains.ki,
            Kd=gains.kd,
            setpoint=0.0,
            output_limits=output_limits,
            auto_mode=True
        )
        self.pid.sample_time = None  # We'll call it every dt
        self.current_error = 0.0
        
    def set_control_mode(self, mode: Literal['position', 'velocity', 'torque']):
        """Set the control mode."""
        self.control_mode = mode
        self.reset()
        
    def set_setpoint(self, setpoint: float):
        """Set the target value for the controller."""
        self.pid.setpoint = setpoint
        
    def compute(self, measurement: float, dt: float) -> float:
        """
        Compute the control output.
        Args:
            measurement: Current measured value
            dt: Time step (s)
        Returns:
            Control output (normalized to output_limits)
        """
        output = self.pid(measurement)
        self.current_error = self.pid.setpoint - measurement
        return output
    
    def get_error(self) -> float:
        """Get the current error."""
        return self.current_error
    
    def reset(self):
        """Reset the controller state."""
        self.pid.reset()
        self.current_error = 0.0
    
    def update_gains(self, gains: PIDGains):
        """Update the PID gains."""
        self.pid.tunings = (gains.kp, gains.ki, gains.kd) 