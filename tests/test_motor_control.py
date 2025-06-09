"""
Test suite for motor control functionality.
"""

import pytest
import numpy as np
from simulation.motor import MotorSimulator, MotorParameters
from controllers.pid_controller import MotorController, PIDGains
from daq.data_logger import DataLogger, MotorTestMetadata
from datetime import datetime

@pytest.fixture
def motor_params():
    """Create motor parameters for testing."""
    return MotorParameters(
        inertia=0.05,  # kg⋅m²
        damping=1.0,   # N⋅m⋅s/rad - Increased damping for stability
        torque_constant=0.1,  # N⋅m/A
        max_torque=4.0,  # N⋅m
        max_speed=20.0,  # rad/s
        resistance=1.0,  # Ω
        inductance=0.1   # H
    )

@pytest.fixture
def pid_gains():
    """Create PID gains for testing."""
    return PIDGains(
        kp=5.0,   # Reduced from 40.0
        ki=0.2,   # Reduced from 8.0
        kd=0.5    # Reduced from 2.0
    )

def test_motor_simulation(motor_params):
    """Test basic motor simulation."""
    motor = MotorSimulator(motor_params)
    
    # Test step response
    dt = 0.001  # 1ms time step
    voltage = 1.0  # 1V input
    load_torque = 0.0  # No load
    
    # Simulate for 1 second
    for _ in range(1000):
        pos, vel, curr = motor.step(dt, voltage)
        
    # Check final values
    assert abs(pos) > 0  # Should have moved
    assert abs(vel) > 0  # Should have velocity
    assert abs(curr) > 0  # Should have current

def test_pid_control(motor_params, pid_gains):
    """Test PID control of motor."""
    motor = MotorSimulator(motor_params)
    controller = MotorController(pid_gains, control_mode='position', output_limits=(-48.0, 48.0))
    
    setpoint = 1.0  # 1 radian
    controller.set_setpoint(setpoint)
    
    dt = 0.001  # 1ms time step
    position_history = []
    velocity_history = []
    error_history = []
    control_history = []
    
    for i in range(10000):  # 10 seconds
        state = motor.get_state()
        pos = state['position']
        vel = state['velocity']
        current = state.get('current', None)
        torque = getattr(motor, 'torque', None)
        position_history.append(pos)
        velocity_history.append(vel)
        control = controller.compute(pos, dt)
        control_history.append(control)
        error_history.append(controller.get_error())
        if i < 100 or i % 1000 == 0:
            print(f"Step {i}: setpoint={setpoint:.3f}, pos={pos:.3f}, vel={vel:.3f}, control={control:.3f}, error={controller.get_error():.3f}, current={current if current is not None else 'N/A'}, torque={torque if torque is not None else 'N/A'}")
        motor.step(dt, control)

    final_pos = motor.get_state()['position']
    max_error = max(abs(e) for e in error_history)
    try:
        settling_time = next(i * dt for i, e in enumerate(error_history) if abs(e) < 0.1)
    except StopIteration:
        settling_time = float('inf')
    overshoot = (max(position_history) - setpoint) / setpoint if max(position_history) > setpoint else 0
    print(f"Final position: {final_pos:.3f}")
    print(f"Maximum error: {max_error:.3f}")
    print(f"Settling time: {settling_time:.3f}s")
    print(f"Overshoot: {overshoot:.1%}")
    print(f"Max position: {max(position_history):.3f}, Min position: {min(position_history):.3f}")
    print(f"Max velocity: {max(velocity_history):.3f}, Min velocity: {min(velocity_history):.3f}")
    print(f"Max control: {max(control_history):.3f}, Min control: {min(control_history):.3f}")
    assert abs(final_pos - setpoint) < 0.5

def test_velocity_pid_control(motor_params, pid_gains):
    """Test PID velocity control of motor."""
    motor = MotorSimulator(motor_params)
    controller = MotorController(pid_gains, control_mode='velocity', output_limits=(-48.0, 48.0))
    
    setpoint = 5.0  # rad/s
    controller.set_setpoint(setpoint)
    
    dt = 0.001  # 1ms time step
    velocity_history = []
    error_history = []
    control_history = []
    
    for i in range(10000):  # 10 seconds
        vel = motor.get_state()['velocity']
        current = motor.get_state().get('current', None)
        torque = getattr(motor, 'torque', None)
        velocity_history.append(vel)
        control = controller.compute(vel, dt)
        control_history.append(control)
        error_history.append(controller.get_error())
        if i < 100 or i % 1000 == 0:
            print(f"Step {i}: setpoint={setpoint:.3f}, vel={vel:.3f}, control={control:.3f}, error={controller.get_error():.3f}, current={current if current is not None else 'N/A'}, torque={torque if torque is not None else 'N/A'}")
        motor.step(dt, control)

    final_vel = motor.get_state()['velocity']
    max_error = max(abs(e) for e in error_history)
    try:
        settling_time = next(i * dt for i, e in enumerate(error_history) if abs(e) < 0.2)
    except StopIteration:
        settling_time = float('inf')
    overshoot = (max(velocity_history) - setpoint) / setpoint if max(velocity_history) > setpoint else 0
    print(f"Final velocity: {final_vel:.3f}")
    print(f"Maximum error: {max_error:.3f}")
    print(f"Settling time: {settling_time:.3f}s")
    print(f"Overshoot: {overshoot:.1%}")
    print(f"Max velocity: {max(velocity_history):.3f}, Min velocity: {min(velocity_history):.3f}")
    print(f"Max control: {max(control_history):.3f}, Min control: {min(control_history):.3f}")
    assert abs(final_vel - setpoint) < 4.0

def test_data_logging(motor_params, pid_gains):
    """Test data logging functionality."""
    motor = MotorSimulator(motor_params)
    controller = MotorController(pid_gains, control_mode='position', output_limits=(-48.0, 48.0))
    logger = DataLogger()
    
    # Create test metadata
    metadata = MotorTestMetadata(
        test_name="test_001",
        timestamp=datetime.now().isoformat(),
        motor_params=motor_params.__dict__,
        pid_gains=pid_gains.__dict__,
        control_mode="position",
        setpoint=1.0,
        duration=10.0,
        sample_rate=1000.0
    )
    
    # Save and reload test data
    logger.start_test(metadata)
    logger.log_data({'position': 0.0, 'velocity': 0.0, 'current': 0.0, 'torque': 0.0})
    logger.end_test()
    loaded = logger.get_test_data(metadata.test_name)
    assert loaded['metadata']['test_name'] == metadata.test_name
    
    # Verify data was saved
    test_data = logger.get_test_data("test_001")
    assert len(test_data['data']) > 0
    assert test_data['metadata']['test_name'] == "test_001" 