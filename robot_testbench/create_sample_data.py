#!/usr/bin/env python3
"""
Create sample test data for the RobotTestBench dashboard.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

def create_sample_test_data():
    """Create a sample test data set with realistic motor control behavior."""
    # Test parameters
    duration = 10.0  # seconds
    sample_rate = 1000.0  # Hz
    dt = 1.0 / sample_rate
    n_samples = int(duration * sample_rate)
    
    # Time array
    time = np.arange(0, duration, dt)
    
    # Create position data (step response with overshoot)
    setpoint = 1.0
    wn = 2.0  # natural frequency
    zeta = 0.5  # damping ratio
    position = setpoint * (1 - np.exp(-zeta * wn * time) * 
                         (np.cos(wn * np.sqrt(1 - zeta**2) * time) + 
                          zeta / np.sqrt(1 - zeta**2) * 
                          np.sin(wn * np.sqrt(1 - zeta**2) * time)))
    
    # Calculate velocity (derivative of position)
    velocity = np.gradient(position, dt)
    
    # Calculate torque (proportional to acceleration + damping)
    acceleration = np.gradient(velocity, dt)
    damping = 0.1
    torque = 0.05 * acceleration + damping * velocity
    
    # Calculate current (proportional to torque)
    torque_constant = 0.1
    current = torque / torque_constant
    
    # Add some noise to measurements
    position_noise = np.random.normal(0, 0.01, n_samples)
    velocity_noise = np.random.normal(0, 0.1, n_samples)
    current_noise = np.random.normal(0, 0.05, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': [datetime.now().isoformat() for _ in range(n_samples)],
        'elapsed_time': time,
        'raw_position': position + position_noise,
        'raw_velocity': velocity + velocity_noise,
        'raw_current': current + current_noise,
        'filtered_position': position,
        'filtered_velocity': velocity,
        'filtered_current': current,
        'torque': torque,
        'control_output': torque,  # In this simple model, control output equals torque
        'error': setpoint - position
    })
    
    # Create metadata
    metadata = {
        "test_name": "sample_test",
        "timestamp": datetime.now().isoformat(),
        "motor_params": {
            "inertia": 0.05,
            "damping": 0.1,
            "torque_constant": 0.1,
            "max_torque": 4.0,
            "max_speed": 20.0,
            "resistance": 1.0,
            "inductance": 0.1,
            "thermal_mass": 0.1,
            "thermal_resistance": 0.5,
            "gear_ratio": 1.0,
            "gear_efficiency": 1.0,
            "ambient_temp": 25.0,
            "max_temp": 100.0,
            "temp_coeff": 0.0039
        },
        "pid_gains": {
            "kp": 5.0,
            "ki": 0.2,
            "kd": 0.5
        },
        "control_mode": "position",
        "setpoint": setpoint,
        "duration": duration,
        "sample_rate": sample_rate,
        "sensor_params": {
            "position_noise": 0.01,
            "velocity_noise": 0.1,
            "current_noise": 0.05
        },
        "motor_specs": {
            "type": "DC Motor",
            "model": "Sample Test Motor",
            "rated_power": 50,
            "rated_torque": 0.5,
            "rated_speed": 3000,
            "rated_current": 2.0,
            "rated_voltage": 24,
            "encoder_resolution": 1024,
            "gear_ratio": 1.0
        },
        "sensor_specs": {
            "position": {
                "type": "Quadrature Encoder",
                "resolution": 0.006,
                "range": 6.28
            },
            "velocity": {
                "type": "Derived from Encoder",
                "resolution": 0.1,
                "range": 50
            },
            "current": {
                "type": "Hall Effect Current Sensor",
                "resolution": 0.01,
                "range": 5
            },
            "torque": {
                "type": "Strain Gauge Load Cell",
                "resolution": 0.01,
                "range": 10
            }
        }
    }
    
    # Create test directory
    test_dir = Path("data/logs/sample_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    data.to_csv(test_dir / "data.csv", index=False)
    
    # Save metadata
    with open(test_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Created sample test data in {test_dir}")

if __name__ == "__main__":
    create_sample_test_data() 