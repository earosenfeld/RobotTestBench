# Robot Test Bench

A comprehensive simulation framework for testing and validating robot control systems, with a focus on realistic sensor modeling and motor control.

## Features

- Realistic motor simulation with physical parameters
- Advanced sensor models:
  - Quadrature encoders with noise and edge triggering
  - Force/torque sensors with drift and hysteresis
  - Joint angle sensors with backlash and limit stops
- PID control system with anti-windup
- Real-time visualization and analysis tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/robot_testbench.git
cd robot_testbench
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Running Tests

```bash
python -m pytest tests/
```

### Running Examples

```bash
python examples/sensor_demo.py
```

### Basic Usage Example

```python
from robot_testbench.sensors import (
    QuadratureEncoder, QuadratureEncoderConfig,
    ForceTorqueSensor, ForceTorqueSensorConfig,
    JointAngleSensor, JointAngleSensorConfig
)

# Create a quadrature encoder
encoder_config = QuadratureEncoderConfig(
    resolution=1000,  # 1000 counts per revolution
    noise_std=0.5,    # Add some noise
    edge_trigger_noise=0.0001  # 100 μs timing noise
)
encoder = QuadratureEncoder(encoder_config)

# Create a force/torque sensor
ft_config = ForceTorqueSensorConfig(
    sensitivity=1.0,  # 1 N⋅m/V
    noise_std=0.1,    # 100 mV noise
    drift_rate=0.01   # 10 mV/s drift
)
ft_sensor = ForceTorqueSensor(ft_config)

# Create a joint angle sensor
ja_config = JointAngleSensorConfig(
    resolution=0.001,  # 1 mrad resolution
    noise_std=0.0005,  # 0.5 mrad noise
    backlash=0.01      # 10 mrad backlash
)
ja_sensor = JointAngleSensor(ja_config)

# Use the sensors
position = 1.0  # radians
velocity = 0.5  # rad/s
force = 10.0    # N⋅m

# Get encoder readings
a, b = encoder.update(position, velocity, 0.01)

# Get force/torque reading
ft_reading = ft_sensor.update(force, 0.01)

# Get joint angle reading
angle_reading = ja_sensor.update(position, velocity)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.