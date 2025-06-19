"""
Dyno simulation module for RobotTestBench.
Implements a motor driving a load motor (dyno) with back EMF and reactive torque.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from .motor_simulator import MotorSimulator, MotorParameters
from robot_testbench.sensors import (
    EncoderSimulator, EncoderConfig,
    ForceTorqueSensor, ForceTorqueSensorConfig,
    JointAngleSensor, JointAngleSensorConfig,
    DAQSimulator
)

@dataclass
class DynoParameters:
    """Parameters for the dyno test setup."""
    drive_motor: MotorParameters  # Parameters for the drive motor
    load_motor: MotorParameters   # Parameters for the load motor
    coupling_stiffness: float = 1000.0  # N⋅m/rad - Stiffness of the coupling
    coupling_damping: float = 0.1      # N⋅m⋅s/rad - Damping of the coupling
    max_torque_transfer: float = 10.0  # N⋅m - Maximum torque that can be transferred
    # Optionally, add more for advanced coupling (backlash, friction, etc.)

class DynoSimulator:
    """
    Simulates a motor driving a load motor (dyno) with back EMF, reactive torque, and sensor/DAQ integration.
    Supports independent drive/load voltages, attachable sensors, DAQ, and fault injection.
    """
    def __init__(self, params: DynoParameters,
                 drive_encoder: Optional[EncoderSimulator] = None,
                 load_encoder: Optional[EncoderSimulator] = None,
                 drive_torque_sensor: Optional[ForceTorqueSensor] = None,
                 load_torque_sensor: Optional[ForceTorqueSensor] = None,
                 drive_angle_sensor: Optional[JointAngleSensor] = None,
                 load_angle_sensor: Optional[JointAngleSensor] = None,
                 daq: Optional[DAQSimulator] = None):
        self.params = params
        self.drive_motor = MotorSimulator(params.drive_motor)
        self.load_motor = MotorSimulator(params.load_motor)
        self.coupling_angle = 0.0  # Relative angle between motors
        self.coupling_velocity = 0.0  # Relative velocity between motors
        # Sensors
        self.drive_encoder = drive_encoder
        self.load_encoder = load_encoder
        self.drive_torque_sensor = drive_torque_sensor
        self.load_torque_sensor = load_torque_sensor
        self.drive_angle_sensor = drive_angle_sensor
        self.load_angle_sensor = load_angle_sensor
        # DAQ
        self.daq = daq
        # Faults
        self.faults = set()
        # Logging
        self.log_data = []
        
    def get_state(self) -> Dict[str, Any]:
        drive_state = self.drive_motor.get_state()
        load_state = self.load_motor.get_state()
        return {
            'drive_position': drive_state['position'],
            'drive_velocity': drive_state['velocity'],
            'drive_current': drive_state['current'],
            'drive_torque': drive_state['torque'],
            'drive_temperature': drive_state['temperature'],
            'load_position': load_state['position'],
            'load_velocity': load_state['velocity'],
            'load_current': load_state['current'],
            'load_torque': load_state['torque'],
            'load_temperature': load_state['temperature'],
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
    
    def step(self, dt: float, drive_voltage: float, load_voltage: float = 0.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step the dyno simulation forward in time.
        Args:
            dt: Time step (s)
            drive_voltage: Voltage applied to drive motor
            load_voltage: Voltage applied to load motor (for regenerative/motoring)
        Returns:
            Tuple of (drive_motor_state, load_motor_state)
        """
        # Fault: mechanical decoupling
        if 'decoupled' in self.faults:
            reactive_torque = 0.0
        else:
            reactive_torque = self._calculate_reactive_torque()
        # Step drive motor with reactive torque as load
        drive_pos, drive_vel, drive_curr = self.drive_motor.step(dt, drive_voltage)
        # Step load motor with reactive torque and back EMF
        load_pos, load_vel, load_curr = self.load_motor.step(dt, load_voltage)
        # Update coupling state
        self.coupling_angle = drive_pos - load_pos
        self.coupling_velocity = drive_vel - load_vel
        # Sensor updates
        sensor_readings = self.get_sensor_readings(dt)
        # DAQ
        if self.daq:
            daq_readings = self.daq.step(sensor_readings, dt)
        else:
            daq_readings = sensor_readings
        # Logging
        self.log(drive_state=self.drive_motor.get_state(), load_state=self.load_motor.get_state(), sensors=daq_readings)
        return self.drive_motor.get_state(), self.load_motor.get_state()
    
    def get_sensor_readings(self, dt: float = 0.0) -> Dict[str, Any]:
        """Return simulated sensor readings for all attached sensors."""
        readings = {}
        drive_state = self.drive_motor.get_state()
        load_state = self.load_motor.get_state()
        if self.drive_encoder:
            readings['drive_encoder'] = self.drive_encoder.update(drive_state['position'], drive_state['velocity'], dt)
        if self.load_encoder:
            readings['load_encoder'] = self.load_encoder.update(load_state['position'], load_state['velocity'], dt)
        if self.drive_torque_sensor:
            readings['drive_torque_sensor'] = self.drive_torque_sensor.update(drive_state['torque'], dt)
        if self.load_torque_sensor:
            readings['load_torque_sensor'] = self.load_torque_sensor.update(load_state['torque'], dt)
        if self.drive_angle_sensor:
            readings['drive_angle_sensor'] = self.drive_angle_sensor.update(drive_state['position'], drive_state['velocity'])
        if self.load_angle_sensor:
            readings['load_angle_sensor'] = self.load_angle_sensor.update(load_state['position'], load_state['velocity'])
        return readings
    
    def inject_fault(self, fault_type: str):
        """Inject a fault (e.g., 'decoupled', 'sensor_dropout', 'motor_fault')."""
        self.faults.add(fault_type)
    
    def clear_fault(self, fault_type: str):
        """Clear a previously injected fault."""
        self.faults.discard(fault_type)
    
    def log(self, drive_state=None, load_state=None, sensors=None):
        """Log the current state and sensor readings."""
        entry = {
            'drive_state': drive_state or self.drive_motor.get_state(),
            'load_state': load_state or self.load_motor.get_state(),
            'sensors': sensors or self.get_sensor_readings()
        }
        self.log_data.append(entry)
    
    def get_log(self):
        """Return the log as a list of dicts."""
        return self.log_data
    
    def reset(self):
        """Reset both motors and coupling state."""
        self.drive_motor.reset()
        self.load_motor.reset()
        self.coupling_angle = 0.0
        self.coupling_velocity = 0.0 
        self.faults.clear()
        self.log_data = []
        # Reset sensors if present
        for sensor in [self.drive_encoder, self.load_encoder, self.drive_torque_sensor, self.load_torque_sensor, self.drive_angle_sensor, self.load_angle_sensor]:
            if sensor and hasattr(sensor, 'reset'):
                sensor.reset() 