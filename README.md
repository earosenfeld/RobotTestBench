# RobotTestBench

A modular Python application for actuator and sensor testing, performance logging, and real-time data visualization. RobotTestBench provides tools for simulating and interfacing with robotic hardware, enabling comprehensive testing and analysis of motor behavior, control loop performance, and system health.

## Features

- **Actuator Control Interface**
  - Simulated motor control
  - PID-based control for torque, velocity, and position modes
  - Optional serial/CAN interface for real hardware

- **Data Acquisition (DAQ) Layer**
  - Real-time sensor data streaming
  - CSV/JSON logging with metadata
  - Test run tagging and organization

- **Dyno Simulator**
  - Robotic joint simulation with configurable parameters
  - Inertia, friction, and load modeling
  - Virtual sensor data generation

- **Real-Time Visualization**
  - Live signal plotting
  - Test limit monitoring
  - Anomaly detection

- **Post-Test Analytics**
  - Test log analysis
  - Performance metrics computation
  - Report generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RobotTestBench.git
cd RobotTestBench
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
RobotTestBench/
├── controllers/       # PID and other control methods
├── daq/              # Data acquisition and logging
├── simulation/       # Dyno and actuator models
├── analytics/        # Test metric computation + reports
├── visualization/    # Real-time and post-test plots
├── tests/            # Unit tests for each module
├── data/             # Saved logs and test cases
├── main.py           # CLI entry point or Dash app launcher
└── README.md
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Configure test parameters through the interface
3. Run tests and monitor real-time data
4. Analyze results and generate reports

## Development

- Python 3.10+ required
- Key dependencies:
  - numpy
  - pandas
  - matplotlib
  - plotly
  - scipy
  - simple-pid
  - pyserial (optional)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details