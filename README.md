# RobotTestBench: Advanced Actuator and Sensor Testing Framework

A comprehensive simulation and testing framework for robot control systems, featuring realistic motor, sensor, and control system models. Designed for rigorous hardware testing and validation of actuators and sensors in humanoid robotics applications.

## Overview

RobotTestBench provides a complete environment for developing, testing, and validating robot control systems. It simulates motors, sensors, and control systems with realistic physical models, allowing you to test control algorithms before deploying them on physical hardware. The framework is particularly well-suited for testing humanoid robot actuators and sensors, with features specifically designed for:

- Actuator performance characterization
- Force/torque sensor validation
- Position/velocity sensor testing
- Dynamic response analysis
- Thermal performance evaluation
- Endurance testing simulation
- Safety limit verification

## Key Features

### 1. Advanced Motor Simulation
The motor simulation models a DC motor with realistic physical characteristics:
- Inertia and damping effects
- Torque and speed limits
- Electrical characteristics (resistance, inductance)
- Back EMF effects
- Thermal modeling
- Efficiency mapping
- Gearbox modeling
- Friction compensation

```python
motor_params = MotorParameters(
    inertia=0.05,      # kg⋅m²
    damping=1.0,       # N⋅m⋅s/rad
    torque_constant=0.1,  # N⋅m/A
    max_torque=4.0,    # N⋅m
    max_speed=20.0,    # rad/s
    resistance=1.0,    # Ω
    inductance=0.1,    # H
    thermal_mass=0.1,  # kg
    thermal_resistance=0.5,  # K/W
    gear_ratio=10.0,   # -
    gear_efficiency=0.95  # -
)
```

#### Technical Details
The motor simulation uses the following differential equations:
```
τ = Kt * i - b * ω - J * dω/dt
v = R * i + L * di/dt + Ke * ω
dT/dt = (P_loss - (T - T_amb)/R_th)/C_th
```
Where:
- τ: Motor torque
- Kt: Torque constant
- i: Current
- b: Damping coefficient
- ω: Angular velocity
- J: Moment of inertia
- v: Applied voltage
- R: Resistance
- L: Inductance
- Ke: Back EMF constant
- T: Temperature
- T_amb: Ambient temperature
- R_th: Thermal resistance
- C_th: Thermal capacitance
- P_loss: Power losses

### 2. Comprehensive Sensor Models

#### Force/Torque Sensor
Advanced strain gauge-based sensor simulation with:
- Temperature-dependent drift
- Hysteresis effects
- Calibration offsets
- Configurable sensitivity
- Cross-axis coupling
- Dynamic response
- Overload protection
- Temperature compensation

```python
ft_config = ForceTorqueSensorConfig(
    sensitivity=1.0,    # N⋅m/V
    noise_std=0.05,     # V
    drift_rate=0.005,   # V/s
    hysteresis=0.05,    # N⋅m
    cross_coupling=0.01,  # N⋅m/N⋅m
    bandwidth=1000.0,   # Hz
    overload_limit=10.0,  # N⋅m
    temp_coeff=0.001    # V/°C
)
```

#### Position/Velocity Sensors
High-precision position and velocity sensing with:
- Multi-turn absolute encoding
- High-speed quadrature counting
- Temperature compensation
- Vibration immunity
- EMI shielding
- Redundancy options

```python
encoder_config = QuadratureEncoderConfig(
    counts_per_rev=1024,  # Resolution
    edge_trigger_noise=0.0001,  # 100 μs
    max_frequency=1000.0,  # Hz
    temp_coeff=0.0001,  # counts/°C
    vibration_immunity=5.0,  # g
    redundancy_mode='dual'  # single/dual/triple
)
```

### 3. Advanced Control System
The control system implements sophisticated control algorithms:
- PID control with anti-windup
- Feedforward compensation
- Adaptive control
- Impedance control
- Force control
- Position control
- Velocity control
- Torque control

```python
# Example advanced control configuration
control_config = ControlConfig(
    # PID gains
    Kp=100.0,  # Proportional gain
    Ki=10.0,   # Integral gain
    Kd=1.0,    # Derivative gain
    
    # Feedforward
    Kff=0.1,   # Feedforward gain
    inertia_ff=True,  # Inertia feedforward
    friction_ff=True,  # Friction feedforward
    
    # Adaptive control
    adaptive_gains=True,  # Enable adaptive gains
    learning_rate=0.01,   # Adaptation rate
    
    # Safety limits
    max_torque=4.0,      # N⋅m
    max_velocity=20.0,   # rad/s
    max_accel=100.0      # rad/s²
)
```

### 4. Real-time Visualization and Analysis

#### Dashboard Features
1. **Interactive Plots**
   - Zoom and pan capabilities
   - Data point inspection
   - Time window selection
   - Multiple y-axis scales
   - FFT analysis
   - Bode plots
   - Nyquist plots
   - Step response analysis

2. **Real-time Updates**
   - Configurable update rate
   - Data buffering
   - Smooth animations
   - Performance metrics
   - System identification
   - Parameter estimation
   - State estimation

3. **Data Export and Analysis**
   - CSV export
   - Plot image saving
   - Data logging
   - Configuration saving
   - Statistical analysis
   - Performance metrics
   - Test report generation
   - Compliance documentation

### 5. Test Automation and Validation

#### Automated Test Suites
- Performance characterization
- Endurance testing
- Environmental testing
- Safety validation
- Compliance testing
- Regression testing
- Integration testing

```python
# Example test suite configuration
test_suite = TestSuite(
    # Performance tests
    performance_tests=[
        'step_response',
        'frequency_response',
        'torque_ripple',
        'efficiency_map'
    ],
    
    # Endurance tests
    endurance_tests=[
        'continuous_operation',
        'thermal_cycling',
        'load_cycling',
        'start_stop_cycles'
    ],
    
    # Environmental tests
    environmental_tests=[
        'temperature_range',
        'vibration',
        'shock',
        'humidity'
    ],
    
    # Safety tests
    safety_tests=[
        'overload_protection',
        'overtemperature',
        'emergency_stop',
        'fault_handling'
    ]
)
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/robot_testbench.git
cd robot_testbench
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

### Running Tests

Run the comprehensive test suite:
```bash
python -m pytest tests/
```

### Example Usage

1. Create a motor simulation:
```python
from simulation.motor import MotorParameters, MotorSimulator

# Define motor parameters
motor_params = MotorParameters(
    inertia=0.05,
    damping=1.0,
    torque_constant=0.1,
    max_torque=4.0,
    max_speed=20.0,
    resistance=1.0,
    inductance=0.1,
    thermal_mass=0.1,
    thermal_resistance=0.5,
    gear_ratio=10.0,
    gear_efficiency=0.95
)

# Create motor simulator
motor = MotorSimulator(motor_params)
```

2. Set up sensors:
```python
from robot_testbench.sensors import (
    QuadratureEncoder, QuadratureEncoderConfig,
    ForceTorqueSensor, ForceTorqueSensorConfig,
    JointAngleSensor, JointAngleSensorConfig
)

# Create sensor configurations
encoder = QuadratureEncoder(QuadratureEncoderConfig())
ft_sensor = ForceTorqueSensor(ForceTorqueSensorConfig())
ja_sensor = JointAngleSensor(JointAngleSensorConfig())
```

3. Run a control loop:
```python
# Set control parameters
setpoint = 1.0  # rad
dt = 0.001      # s

# Control loop
for _ in range(1000):
    # Get sensor readings
    position = encoder.get_position()
    velocity = encoder.get_velocity()
    torque = ft_sensor.get_torque()
    
    # Compute control output
    error = setpoint - position
    control_output = pid_controller.compute(error, dt)
    
    # Apply control
    motor.set_torque(control_output)
    
    # Step simulation
    motor.step(dt)
    
    # Log data
    logger.debug("Position: %f, Velocity: %f, Torque: %f",
                position, velocity, torque)
```

## Best Practices

### 1. Motor Control
- Start with conservative PID gains
- Use anti-windup protection
- Implement proper error handling
- Monitor for saturation
- Log control signals
- Consider thermal effects
- Implement safety limits
- Use feedforward control

### 2. Sensor Usage
- Calibrate sensors before use
- Handle sensor noise appropriately
- Consider temperature effects
- Implement proper filtering
- Monitor sensor health
- Use redundant sensors
- Implement sensor fusion
- Validate sensor data

### 3. Simulation
- Use appropriate time steps
- Validate physical parameters
- Monitor energy conservation
- Check for numerical stability
- Log simulation state
- Consider real-time constraints
- Implement proper initialization
- Handle discontinuities

### 4. Visualization
- Use appropriate scales
- Include units in plots
- Add clear labels
- Implement proper legends
- Save important data
- Generate test reports
- Create performance plots
- Document test results

## Troubleshooting Guide

### Common Issues

1. **Motor Instability**
   - Check PID gains
   - Verify motor parameters
   - Monitor control output
   - Check for saturation
   - Review anti-windup
   - Check thermal limits
   - Verify safety limits
   - Review feedforward

2. **Sensor Noise**
   - Verify sensor configuration
   - Check sampling rate
   - Implement filtering
   - Review calibration
   - Check connections
   - Consider EMI
   - Check grounding
   - Review shielding

3. **Simulation Issues**
   - Verify time step
   - Check initial conditions
   - Review physical parameters
   - Monitor numerical stability
   - Check for NaN values
   - Verify constraints
   - Check discretization
   - Review integration

4. **Visualization Problems**
   - Check update rate
   - Verify data format
   - Review plot settings
   - Check memory usage
   - Monitor performance
   - Check data buffering
   - Verify export format
   - Review plot limits

### Debugging Tips

1. **Logging**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use in code
logger.debug("Motor position: %f", position)
logger.info("Control output: %f", control_output)
logger.warning("Sensor noise high: %f", noise_level)
logger.error("Simulation error: %s", error_msg)
```

2. **Performance Monitoring**
```python
import time

start_time = time.time()
# ... your code ...
elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.3f} seconds")
```

3. **Data Validation**
```python
def validate_motor_state(position, velocity, torque):
    if not (0 <= position <= 2*np.pi):
        raise ValueError("Position out of range")
    if abs(velocity) > max_speed:
        raise ValueError("Velocity exceeds limit")
    if abs(torque) > max_torque:
        raise ValueError("Torque exceeds limit")
```

## Project Structure

```
robot_testbench/
├── simulation/
│   ├── motor.py          # Motor simulation
│   ├── sensors.py        # Sensor simulation
│   └── dyno.py          # Dynamometer simulation
├── robot_testbench/
│   ├── sensors.py        # Sensor implementations
│   └── control.py        # Control system
├── tests/
│   ├── test_motor_control.py
│   └── test_sensors.py
├── simple_dashboard.py   # Visualization dashboard
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. **Code Style**
   - Follow PEP 8
   - Use type hints
   - Write docstrings
   - Add comments
   - Keep it simple

2. **Testing**
   - Write unit tests
   - Add integration tests
   - Test edge cases
   - Verify performance
   - Check coverage

3. **Documentation**
   - Update README
   - Add docstrings
   - Include examples
   - Document changes
   - Add comments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Inspired by real-world robotics challenges and control system needs
- Developed with a focus on humanoid robotics applications
- Designed for rigorous hardware testing and validation