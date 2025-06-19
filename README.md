# Robot Test Bench

A comprehensive testing framework for robotic systems, providing tools for motor control simulation, sensor emulation, data acquisition, and real-time visualization.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/robot_testbench.git
cd robot_testbench

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install the package in development mode
pip install -e .
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test types
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration

# Run with coverage
pytest --cov=robot_testbench --cov-report=term-missing
```

### 3. Generate Sample Data

```bash
# Create sample test data for the dashboard
python -m robot_testbench.create_sample_data
python -m robot_testbench.create_harmonic_drive_data
```

### 4. Run Dashboard

```bash
# Start the main dashboard (opens in browser at http://localhost:8050)
python -m robot_testbench.run_test_dashboard

# Run on different port
python -m robot_testbench.run_test_dashboard --port 8080

# Run in debug mode
python -m robot_testbench.run_test_dashboard --debug
```

### 5. Run Examples

```bash
# Simple dashboard example
python examples/simple_dashboard.py

# Sensor demonstration
python examples/sensor_demo.py

# Electrical simulation
python examples/electrical_sim_example.py
```

## What You Get

### 🧪 **Testing Framework**
- **Unit Tests**: Individual component testing (motors, sensors, controllers)
- **Integration Tests**: System-level testing with multiple components
- **Performance Tests**: Benchmarking and performance analysis
- **Test Protocols**: YAML-based test configurations

### 📊 **Dashboard & Visualization**
- **Real-time Dashboard**: Web-based interface at `http://localhost:8050`
- **Test Results**: View and analyze test data with SPC overlays
- **Motor Simulations**: Real-time motor control visualization
- **Sensor Data**: Encoder, force/torque, and joint angle sensor readings

### 🔧 **Core Components**
- **Motor Simulation**: Realistic motor models with electrical and mechanical dynamics
- **Sensor Emulation**: Encoder, force/torque, and joint angle sensor simulation
- **Control Systems**: PID controllers with anti-windup and feedforward
- **Data Acquisition**: High-speed data logging and signal processing

## Project Structure

```
RobotTestBench/
├── robot_testbench/          # Main package
│   ├── motor/               # Motor simulation
│   ├── sensors/             # Sensor emulation
│   ├── control/             # Control systems
│   ├── dashboard/           # Visualization
│   └── visualization/       # Plot utilities
├── examples/                # Example scripts
├── tests/                   # Test suite
├── data/logs/               # Test data storage
└── reports/                 # Test results
```

## Common Commands

### Testing
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --benchmark-only   # Performance benchmarks
python tests/run_tests.py --coverage --html  # Coverage report
```

### Dashboard
```bash
python -m robot_testbench.run_test_dashboard     # Main dashboard
python examples/simple_dashboard.py              # Simple example
python examples/sensor_demo.py                   # Sensor demo
```

### Data Generation
```bash
python -m robot_testbench.create_sample_data     # Basic test data
python -m robot_testbench.create_harmonic_drive_data  # Advanced test data
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError: No module named 'robot_testbench'`:
```bash
pip install -e .  # Install in development mode
```

### Missing Dependencies
If you get missing module errors:
```bash
pip install -r requirements.txt  # Install all dependencies
```

### Dashboard Not Loading
Make sure you have sample data:
```bash
python -m robot_testbench.create_sample_data
python -m robot_testbench.run_test_dashboard
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black .
isort .
pylint robot_testbench
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
