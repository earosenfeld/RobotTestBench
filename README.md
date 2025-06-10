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

# RobotTestBench Dashboard

## Overview

**RobotTestBench** is a modular Python package for simulating, testing, and analyzing motor control systems, with a focus on robotics and humanoid actuators. It provides:
- A professional web dashboard for visualizing test results and analytics
- Support for multiple test scenarios (motors, sensors, actuators)
- Statistical Process Control (SPC) overlays and advanced analysis
- Easy extensibility for new hardware and test profiles

## Features
- **Modern Dash/Plotly web dashboard** for interactive analysis
- **SPC overlays** (mean, ±3σ, out-of-control points) on all key signal plots
- **Thermal, power, and efficiency analysis** with control limits
- **Hardware and sensor configuration tables**
- **Comparison and A/B testing** between multiple test runs
- **Support for custom test scenarios** (e.g., BLDC, Harmonic Drive actuators)

## Directory Structure
```
robot_testbench/
  visualization/
    test_results.py      # Main dashboard code
  create_sample_data.py  # Script to generate sample BLDC test data
  create_harmonic_drive_data.py # Script to generate Harmonic Drive test data
  ...
data/
  logs/
    BLDC_HallEffect/
      metadata.json
      data.csv
    HarmonicDrive_HumanoidJoint/
      metadata.json
      data.csv
    ...
```

## Test Data Structure
Each test scenario has its own directory under `data/logs/`, containing:
- `metadata.json`: Hardware configuration, test parameters, sensor specs, etc.
- `data.csv`: Time-series data (raw/filtered position, velocity, current, torque, etc.)

### Example `metadata.json`
```json
{
  "test_name": "BLDC_HallEffect",
  "motor_specs": { "type": "BLDC", ... },
  "sensor_specs": { "position": {"type": "Hall Effect Encoder", ... }, ... },
  ...
}
```

### Example `data.csv` columns
- `timestamp`, `elapsed_time`
- `raw_position`, `filtered_position`
- `raw_velocity`, `filtered_velocity`
- `raw_current`, `filtered_current`
- `torque`

## Setup & Usage

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Generate sample data (optional):**
   ```sh
   python -m robot_testbench.create_sample_data
   python -m robot_testbench.create_harmonic_drive_data
   ```

3. **Run the dashboard:**
   ```sh
   python -m robot_testbench.run_test_dashboard
   ```
   The dashboard will be available at [http://localhost:8050](http://localhost:8050)

4. **Select a test** from the dropdown to view detailed analytics, SPC overlays, and hardware configuration.

## Adding a New Test Scenario
1. **Create a new directory** under `data/logs/YourTestName/`.
2. **Add a `metadata.json`** file describing the hardware, sensors, and test parameters.
3. **Add a `data.csv`** file with the required columns (see above).
4. The new test will automatically appear in the dashboard dropdown.

## Analysis Features
- **SPC Control Charts:** Mean, ±3σ, and out-of-control points for all filtered signals
- **Thermal Analysis:** Power dissipation, efficiency, and SPC overlays
- **Hardware/Sensor Tables:** Full configuration for each test
- **Comparison:** Side-by-side and A/B test comparison
- **Export:** Download plots and data for reporting

## Example Test Scenarios
- **BLDC_HallEffect:** Typical BLDC motor with Hall effect encoder
- **HarmonicDrive_HumanoidJoint:** High-torque BLDC with Harmonic Drive gearbox and absolute encoder

## Customization
- Add new test profiles by generating new `metadata.json` and `data.csv` files
- Modify dashboard code in `robot_testbench/visualization/test_results.py` for advanced analytics

## Support
For questions or to contribute new test scenarios, open an issue or pull request.
