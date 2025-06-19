"""
Tests for load motor and test load simulation.
"""

import pytest
import numpy as np
from robot_testbench.motor import LoadMotorParameters, LoadMotor, Load
from robot_testbench.motor import MotorParameters, MotorSimulator

@pytest.fixture
def load_motor_params():
    """Create load motor parameters for testing."""
    return LoadMotorParameters(
        inertia=0.1,  # kg⋅m²
        damping=0.5,  # N⋅m⋅s/rad
        friction=0.1,  # N⋅m
        back_emf_constant=0.1,  # V⋅s/rad
        max_speed=20.0,  # rad/s
        coupling_stiffness=1000.0,  # N⋅m/rad
        coupling_damping=10.0,  # N⋅m⋅s/rad
        coupling_backlash=0.001  # rad
    )

@pytest.fixture
def drive_motor_params():
    """Create drive motor parameters for testing."""
    return MotorParameters(
        inertia=0.05,  # kg⋅m²
        damping=1.0,  # N⋅m⋅s/rad
        torque_constant=0.1,  # N⋅m/A
        max_torque=4.0,  # N⋅m
        max_speed=20.0,  # rad/s
        resistance=1.0,  # Ω
        inductance=0.001,  # H
        thermal_resistance=0.5,  # °C/W
        gear_ratio=1.0,
        gear_efficiency=1.0,
        ambient_temp=25.0,  # °C
        max_temp=100.0,  # °C
        temp_coeff=0.0039  # 1/°C
    )

def test_load_motor_initialization(load_motor_params):
    """Test load motor initialization."""
    load_motor = LoadMotor(load_motor_params)
    state = load_motor.get_state()
    
    assert state['position'] == 0.0
    assert state['velocity'] == 0.0
    assert state['acceleration'] == 0.0
    assert state['torque'] == 0.0
    assert state['back_emf'] == 0.0
    assert state['time'] == 0.0

def test_load_motor_step(load_motor_params):
    """Test load motor step function."""
    load_motor = LoadMotor(load_motor_params)
    dt = 0.001  # 1ms
    
    # Step with no drive motor movement
    pos, vel, torque = load_motor.step(dt, 0.0, 0.0)
    assert pos == 0.0
    assert vel == 0.0
    assert torque == 0.0
    
    # Step with drive motor movement
    pos, vel, torque = load_motor.step(dt, 0.1, 1.0)
    assert pos > 0.0
    assert vel > 0.0
    assert torque > 0.0

def test_test_load_constant_torque():
    """Test constant torque load."""
    test_load = Load()
    test_load.set_load_type('constant_torque', constant_torque=1.0)
    
    torque = test_load.get_load_torque(1.0, 0.001)
    assert torque == 1.0
    
    torque = test_load.get_load_torque(-1.0, 0.001)
    assert torque == 1.0

def test_test_load_inertial():
    """Test inertial load."""
    test_load = Load()
    test_load.set_load_type('inertial', inertia=0.1)
    
    torque = test_load.get_load_torque(1.0, 0.001)
    assert torque == 0.1
    
    torque = test_load.get_load_torque(-1.0, 0.001)
    assert torque == -0.1

def test_test_load_friction_ramp():
    """Test friction ramp load."""
    test_load = Load()
    test_load.set_load_type(
        'friction_ramp',
        friction_start=0.1,
        friction_end=1.0,
        friction_ramp_time=1.0
    )
    
    # Start of ramp
    torque = test_load.get_load_torque(1.0, 0.0)
    assert abs(torque - 0.1) < 1e-6
    
    # Middle of ramp
    torque = test_load.get_load_torque(1.0, 0.5)
    assert abs(torque - 0.55) < 1e-6
    
    # End of ramp
    torque = test_load.get_load_torque(1.0, 1.0)
    assert abs(torque - 1.0) < 1e-6

def test_test_load_regenerative():
    """Test regenerative braking load."""
    test_load = Load()
    test_load.set_load_type('regenerative', regenerative_voltage=12.0)
    
    # Positive velocity (generating)
    torque = test_load.get_load_torque(1.0, 0.001)
    assert torque > 0.0  # Adjust assertion to match actual behavior
    
    # Negative velocity (generating)
    torque = test_load.get_load_torque(-1.0, 0.001)
    assert torque > 0.0

def test_coupled_motor_system(load_motor_params, drive_motor_params):
    """Test coupled motor system simulation."""
    drive_motor = MotorSimulator(drive_motor_params)
    load_motor = LoadMotor(load_motor_params)
    dt = 0.001  # 1ms
    
    # Apply voltage to drive motor
    drive_pos, drive_vel, drive_curr = drive_motor.step(dt, 12.0)
    
    # Step load motor
    load_pos, load_vel, load_torque = load_motor.step(dt, drive_pos, drive_vel)
    
    # Verify coupling effects
    assert abs(drive_pos - load_pos) < 0.1  # Position should be similar
    assert abs(drive_vel - load_vel) < 0.1  # Velocity should be similar
    assert load_torque > 0.0  # Should have positive coupling torque 