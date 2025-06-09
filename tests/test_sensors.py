"""
Test suite for sensor simulation functionality.
"""

import pytest
import numpy as np
from simulation.sensors import SensorSimulator, SensorParameters
from robot_testbench.sensors import (
    QuadratureEncoder, QuadratureEncoderConfig,
    ForceTorqueSensor, ForceTorqueSensorConfig,
    JointAngleSensor, JointAngleSensorConfig
)

@pytest.fixture
def sensor_params():
    """Create sensor parameters for testing."""
    return SensorParameters(
        position_noise_std=0.001,  # rad
        velocity_noise_std=0.01,   # rad/s
        current_noise_std=0.1,     # A
        sampling_rate=1000.0,      # Hz
        filter_type='lowpass',     # Start with lowpass filter
        filter_cutoff=50.0,        # Hz
        kalman_process_noise=0.1,
        kalman_measurement_noise=1.0
    )

def test_noise_injection(sensor_params):
    """Test that noise is properly injected into sensor signals."""
    sensor = SensorSimulator(sensor_params)
    
    # Test multiple samples
    n_samples = 1000
    positions = []
    velocities = []
    currents = []
    
    for _ in range(n_samples):
        raw_pos, raw_vel, raw_curr = sensor.add_noise(1.0, 2.0, 3.0)
        positions.append(raw_pos)
        velocities.append(raw_vel)
        currents.append(raw_curr)
    
    # Convert to numpy arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    currents = np.array(currents)
    
    # Check means (should be close to input values)
    assert abs(np.mean(positions) - 1.0) < 0.1
    assert abs(np.mean(velocities) - 2.0) < 0.1
    assert abs(np.mean(currents) - 3.0) < 0.1
    
    # Check standard deviations (should be close to noise parameters)
    assert abs(np.std(positions) - sensor_params.position_noise_std) < 0.1
    assert abs(np.std(velocities) - sensor_params.velocity_noise_std) < 0.1
    assert abs(np.std(currents) - sensor_params.current_noise_std) < 0.1

def test_lowpass_filtering(sensor_params):
    """Test lowpass filter functionality."""
    sensor = SensorSimulator(sensor_params)
    
    # Generate noisy signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = signal + noise
    
    # Apply filter
    filtered = sensor._apply_lowpass_filter(noisy_signal, sensor.position_window)
    
    # Check that filtered signal has less variance than noisy signal
    assert np.std(filtered) < np.std(noisy_signal)
    
    # Check that filtered signal preserves the mean
    assert np.abs(np.mean(filtered) - np.mean(signal)) < 0.1

def test_kalman_filtering():
    """Test Kalman filter functionality."""
    # Create sensor with Kalman filter
    params = SensorParameters(
        filter_type='kalman',
        kalman_process_noise=0.1,
        kalman_measurement_noise=1.0
    )
    sensor = SensorSimulator(params)
    
    # Generate test data
    n_samples = 100
    true_position = np.linspace(0, 1, n_samples)  # Linear motion
    true_velocity = np.ones(n_samples)  # Constant velocity
    
    # Add noise to measurements
    noisy_position = true_position + np.random.normal(0, 0.1, n_samples)
    
    # Apply Kalman filter
    filtered_positions = []
    filtered_velocities = []
    
    for pos in noisy_position:
        measurement = np.array([pos])
        filtered_state = sensor._apply_kalman_filter(measurement)
        filtered_positions.append(filtered_state[0])
        filtered_velocities.append(filtered_state[1])
    
    # Convert to numpy arrays
    filtered_positions = np.array(filtered_positions)
    filtered_velocities = np.array(filtered_velocities)
    
    # Check that filtered position is closer to true position than noisy
    position_error_noisy = np.mean((noisy_position - true_position) ** 2)
    position_error_filtered = np.mean((filtered_positions - true_position) ** 2)
    assert position_error_filtered < position_error_noisy
    
    # Check that filtered velocity is close to true velocity
    velocity_error = np.mean((filtered_velocities - true_velocity) ** 2)
    assert velocity_error < 1.1

def test_signal_processing(sensor_params):
    """Test complete signal processing pipeline."""
    sensor = SensorSimulator(sensor_params)
    
    # Test multiple samples
    n_samples = 1000
    raw_positions = []
    raw_velocities = []
    raw_currents = []
    filtered_positions = []
    filtered_velocities = []
    filtered_currents = []
    
    for _ in range(n_samples):
        (raw_pos, raw_vel, raw_curr), (filt_pos, filt_vel, filt_curr) = sensor.process_signals(1.0, 2.0, 3.0)
        raw_positions.append(raw_pos)
        raw_velocities.append(raw_vel)
        raw_currents.append(raw_curr)
        filtered_positions.append(filt_pos)
        filtered_velocities.append(filt_vel)
        filtered_currents.append(filt_curr)
    
    # Convert to numpy arrays
    raw_positions = np.array(raw_positions)
    filtered_positions = np.array(filtered_positions)
    
    # Check that filtered signal has less variance than raw
    assert np.std(filtered_positions) < np.std(raw_positions)
    
    # Check that filtered signal still tracks the mean (allowing for small deviations)
    mean_error = abs(np.mean(filtered_positions) - 1.0)
    print('Mean error:', mean_error)
    assert mean_error < 0.1  # Allow for small deviations from input value 

def test_quadrature_encoder_basic():
    """Test basic quadrature encoder functionality."""
    config = QuadratureEncoderConfig(
        resolution=1000,  # 1000 counts per revolution
        noise_std=0.0,    # No noise for basic test
        edge_trigger_noise=0.0
    )
    encoder = QuadratureEncoder(config)
    
    # Test positive rotation
    a, b = encoder.update(0.0, 1.0, 0.01)  # Initial position, positive velocity
    assert a in [0, 1]
    assert b in [0, 1]
    
    # Test negative rotation
    a, b = encoder.update(0.0, -1.0, 0.01)  # Initial position, negative velocity
    assert a in [0, 1]
    assert b in [0, 1]

def test_quadrature_encoder_noise():
    """Test quadrature encoder with noise."""
    config = QuadratureEncoderConfig(
        resolution=1000,
        noise_std=1.0,    # Add some noise
        edge_trigger_noise=0.001  # Add some timing noise
    )
    encoder = QuadratureEncoder(config)
    
    # Test multiple updates
    for _ in range(100):
        a, b = encoder.update(0.0, 1.0, 0.01)
        assert a in [0, 1]
        assert b in [0, 1]

def test_force_torque_sensor_basic():
    """Test basic force/torque sensor functionality."""
    config = ForceTorqueSensorConfig(
        sensitivity=1.0,  # 1 N⋅m/V
        noise_std=0.0,    # No noise for basic test
        drift_rate=0.0,
        hysteresis=0.0
    )
    sensor = ForceTorqueSensor(config)
    
    # Test basic reading
    reading = sensor.update(1.0, 0.01)  # 1 N⋅m
    assert abs(reading - 1.0) < 1e-6

def test_force_torque_sensor_effects():
    """Test force/torque sensor with various effects."""
    config = ForceTorqueSensorConfig(
        sensitivity=1.0,
        noise_std=0.1,    # Add noise
        drift_rate=0.01,  # Add drift
        hysteresis=0.1,   # Add hysteresis
        temperature_coefficient=0.001
    )
    sensor = ForceTorqueSensor(config)
    
    # Test temperature effect
    sensor.set_temperature(50.0)  # 25°C above reference
    reading1 = sensor.update(1.0, 0.01)
    
    sensor.set_temperature(25.0)  # Back to reference
    reading2 = sensor.update(1.0, 0.01)
    
    assert abs(reading1 - reading2) > 0  # Should be different due to temperature

def test_joint_angle_sensor_basic():
    """Test basic joint angle sensor functionality."""
    config = JointAngleSensorConfig(
        resolution=0.001,  # 1 mrad resolution
        noise_std=0.0,     # No noise for basic test
        backlash=0.0
    )
    sensor = JointAngleSensor(config)
    
    # Test basic reading
    reading = sensor.update(1.0, 0.0)  # 1 rad, no velocity
    assert abs(reading - 1.0) < 1e-6

def test_joint_angle_sensor_backlash():
    """Test joint angle sensor with backlash."""
    config = JointAngleSensorConfig(
        resolution=0.001,
        backlash=0.01,  # 10 mrad backlash
        noise_std=0.0
    )
    sensor = JointAngleSensor(config)
    
    # Test direction change
    reading1 = sensor.update(1.0, 1.0)   # Positive velocity
    reading2 = sensor.update(1.0, -1.0)  # Negative velocity
    assert abs(reading1 - reading2) > 0  # Should be different due to backlash

def test_joint_angle_sensor_limits():
    """Test joint angle sensor limit stops."""
    config = JointAngleSensorConfig(
        resolution=0.001,
        limit_stops=(-1.0, 1.0)  # Set limits to ±1 rad
    )
    sensor = JointAngleSensor(config)
    
    # Test within limits
    sensor.update(0.5, 0.0)  # Should work
    
    # Test outside limits
    with pytest.raises(ValueError):
        sensor.update(2.0, 0.0)  # Should raise error

def test_sensor_temperature_effects():
    """Test temperature effects on sensors."""
    # Test force/torque sensor temperature effect
    ft_config = ForceTorqueSensorConfig(
        sensitivity=1.0,
        temperature_coefficient=0.01
    )
    ft_sensor = ForceTorqueSensor(ft_config)
    
    # Test joint angle sensor temperature effect
    ja_config = JointAngleSensorConfig(
        resolution=0.001,
        temperature_coefficient=0.001
    )
    ja_sensor = JointAngleSensor(ja_config)
    
    # Test at different temperatures
    temperatures = [0.0, 25.0, 50.0, 75.0]
    ft_readings = []
    ja_readings = []
    
    for temp in temperatures:
        ft_sensor.set_temperature(temp)
        ja_sensor.set_temperature(temp)
        
        ft_readings.append(ft_sensor.update(1.0, 0.01))
        ja_readings.append(ja_sensor.update(1.0, 0.0))
    
    # Verify readings change with temperature
    assert len(set(ft_readings)) > 1
    assert len(set(ja_readings)) > 1 