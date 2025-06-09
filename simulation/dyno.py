"""
Dyno simulation module for RobotTestBench.
Implements a motor driving a load motor (dyno) with back EMF and reactive torque.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from .motor import MotorSimulator, MotorParameters

@dataclass
class DynoParameters:
    """Parameters for the dyno test setup."""
    drive_motor: MotorParameters  # Parameters for the drive motor
    load_motor: MotorParameters   # Parameters for the load motor
    coupling_stiffness: float = 1000.0  # N⋅m/rad - Stiffness of the coupling
    coupling_damping: float = 0.1      # N⋅m⋅s/rad - Damping of the coupling
    max_torque_transfer: float = 10.0  # N⋅m - Maximum torque that can be transferred

class DynoSimulator:
    """Simulates a motor driving a load motor (dyno) with back EMF and reactive torque."""
    
    def __init__(self, params: DynoParameters):
        """Initialize dyno with parameters."""
        self.params = params
        self.drive_motor = MotorSimulator(params.drive_motor)
        self.load_motor = MotorSimulator(params.load_motor)
        self.coupling_angle = 0.0  # Relative angle between motors
        self.coupling_velocity = 0.0  # Relative velocity between motors
        
    def get_state(self) -> Dict[str, float]:
        """Get current state of both motors and coupling."""
        drive_state = self.drive_motor.get_state()
        load_state = self.load_motor.get_state()
        
        return {
            'drive_position': drive_state['position'],
            'drive_velocity': drive_state['velocity'],
            'drive_current': drive_state['current'],
            'drive_torque': drive_state['torque'],
            'load_position': load_state['position'],
            'load_velocity': load_state['velocity'],
            'load_current': load_state['current'],
            'load_torque': load_state['torque'],
            'coupling_angle': self.coupling_angle,
            'coupling_velocity': self.coupling_velocity,
            'back_emf': self._calculate_back_emf(),
            'reactive_torque': self._calculate_reactive_torque()
        }
    
    def _calculate_back_emf(self) -> float:
        """Calculate back EMF from the load motor."""
        # Back EMF is proportional to the load motor's velocity and its torque constant
        return self.load_motor.velocity * self.params.load_motor.torque_constant
    
    def _calculate_reactive_torque(self) -> float:
        """Calculate reactive torque between motors."""
        # Calculate torque from coupling stiffness and damping
        stiffness_torque = self.coupling_angle * self.params.coupling_stiffness
        damping_torque = self.coupling_velocity * self.params.coupling_damping
        
        # Total reactive torque
        reactive_torque = stiffness_torque + damping_torque
        
        # Limit the maximum torque transfer
        return np.clip(reactive_torque, -self.params.max_torque_transfer, self.params.max_torque_transfer)
    
    def step(self, drive_voltage: float, load_voltage: float, dt: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Step the dyno simulation forward in time.
        
        Args:
            drive_voltage: Voltage applied to drive motor
            load_voltage: Voltage applied to load motor (for regenerative braking)
            dt: Time step (s)
            
        Returns:
            Tuple of (drive_motor_state, load_motor_state)
        """
        # Calculate reactive torque
        reactive_torque = self._calculate_reactive_torque()
        
        # Step drive motor with reactive torque as load
        drive_state = self.drive_motor.step(drive_voltage, -reactive_torque, dt)
        
        # Step load motor with reactive torque and back EMF
        back_emf = self._calculate_back_emf()
        load_state = self.load_motor.step(load_voltage, reactive_torque, dt)
        
        # Update coupling state
        self.coupling_angle = self.drive_motor.position - self.load_motor.position
        self.coupling_velocity = self.drive_motor.velocity - self.load_motor.velocity
        
        return drive_state, load_state
    
    def reset(self):
        """Reset both motors and coupling state."""
        self.drive_motor.reset()
        self.load_motor.reset()
        self.coupling_angle = 0.0
        self.coupling_velocity = 0.0 