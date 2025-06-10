"""
Pytest configuration and common fixtures.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Generator

from robot_testbench.motor import MotorSimulator, MotorParameters
from robot_testbench.sensors import EncoderSimulator, EncoderConfig
from robot_testbench.control import PIDController, PIDConfig

@pytest.fixture
def motor_config() -> MotorParameters:
    """Create a default motor configuration."""
    return MotorParameters(
        inertia=0.001,      # kg⋅m²
        damping=0.001,      # N⋅m⋅s/rad
        torque_constant=0.1,  # N⋅m/A
        max_torque=10.0,    # N⋅m
        max_speed=1000.0,   # rad/s
        resistance=1.0,     # Ω
        inductance=0.001,   # H
        thermal_resistance=0.5,  # K/W
        gear_ratio=10.0,    # -
        gear_efficiency=0.95  # -
    )

@pytest.fixture
def motor(motor_config: MotorParameters) -> MotorSimulator:
    """Create a motor simulator instance."""
    return MotorSimulator(motor_config)

@pytest.fixture
def encoder_config() -> EncoderConfig:
    """Create a default encoder configuration."""
    return EncoderConfig(
        counts_per_rev=1024,  # Resolution
        noise_std=0.1,        # counts
        quantization=True,    # Enable quantization
        max_frequency=1000.0  # Hz
    )

@pytest.fixture
def encoder(encoder_config: EncoderConfig) -> EncoderSimulator:
    """Create an encoder simulator instance."""
    return EncoderSimulator(encoder_config)

@pytest.fixture
def pid_config() -> PIDConfig:
    """Create a default PID controller configuration."""
    return PIDConfig(
        kp=100.0,  # Proportional gain
        ki=10.0,   # Integral gain
        kd=1.0,    # Derivative gain
        output_limits=(-10.0, 10.0),  # N⋅m
        sample_time=0.001  # s
    )

@pytest.fixture
def pid_controller(pid_config: PIDConfig) -> PIDController:
    """Create a PID controller instance."""
    return PIDController(pid_config)

@pytest.fixture
def test_plan() -> Dict:
    """Load the default test plan."""
    plan_path = Path(__file__).parent / "protocols" / "motor_validation.yaml"
    with open(plan_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def test_data() -> Generator[Dict, None, None]:
    """Generate test data for analysis."""
    data = {
        'time': [],
        'position': [],
        'velocity': [],
        'torque': [],
        'setpoint': [],
        'error': []
    }
    yield data 