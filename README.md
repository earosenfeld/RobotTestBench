# Robot Test Bench

A comprehensive testing framework for robotic systems, providing tools for motor control simulation, sensor emulation, data acquisition, and real-time visualization.

## Features

- **Motor Simulation**: Realistic motor models with electrical and mechanical dynamics
- **Sensor Emulation**: Encoder, force/torque, and joint angle sensor simulation
- **Control Systems**: PID controllers with anti-windup and feedforward
- **Data Acquisition**: High-speed data logging and signal processing
- **Visualization**: Real-time dashboards and plot generation
- **Hardware-in-the-Loop**: Interface for future hardware integration
- **Test Automation**: Configurable test plans and performance analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/robot_testbench.git
cd robot_testbench

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

## Quick Start

```python
from robot_testbench import MotorSimulator, PIDController, MotorDashboard

# Create motor simulator
motor = MotorSimulator(
    inertia=0.001,      # kg⋅m²
    damping=0.001,      # N⋅m⋅s/rad
    torque_constant=0.1  # N⋅m/A
)

# Create PID controller
controller = PIDController(
    kp=100.0,  # Proportional gain
    ki=10.0,   # Integral gain
    kd=1.0     # Derivative gain
)

# Create dashboard
dashboard = MotorDashboard()

# Run simulation
time = 0.0
dt = 0.001  # s
duration = 10.0  # s

while time < duration:
    # Get sensor readings
    position = motor.get_position()
    velocity = motor.get_velocity()
    
    # Compute control output
    torque = controller.compute(position, velocity)
    
    # Apply control
    motor.apply_torque(torque)
    
    # Update simulation
    motor.step(dt)
    
    # Update dashboard
    dashboard.update(time, {
        "position": position,
        "velocity": velocity,
        "torque": torque
    })
    
    time += dt

# Save results
dashboard.save("reports/test_run")
```

## Project Structure

```
RobotTestBench/
├── robot_testbench/          # Main package
│   ├── motor/               # Motor simulation
│   ├── sensors/             # Sensor emulation
│   ├── control/             # Control systems
│   ├── daq/                 # Data acquisition
│   ├── dashboard/           # Visualization
│   ├── visualization/       # Plot utilities
│   ├── analytics/           # Performance analysis
│   ├── hil/                 # Hardware interface
│   └── utils/               # Shared utilities
├── examples/                # Example scripts
├── tests/                   # Test suite
├── test_plans/              # Test procedures
├── reports/                 # Test results
└── assets/                  # Resources
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black .
isort .
pylint robot_testbench
mypy robot_testbench
```

## Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
