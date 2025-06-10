from robot_testbench.control import PIDConfig, PIDController

def test_pid_controller_basic():
    """Test basic PID controller functionality."""
    config = PIDConfig(
        kp=1.0,
        ki=0.1,
        kd=0.01
    )
    pid = PIDController(config)
    # ... existing code ...

def test_pid_controller_with_limits():
    """Test PID controller with output and integrator limits."""
    config = PIDConfig(
        kp=2.0,
        ki=0.5,
        kd=0.05
    )
    pid = PIDController(config)
    # ... existing code ...

def test_pid_controller_with_derivative_filter():
    """Test PID controller with derivative filter."""
    config = PIDConfig(
        kp=1.0,
        ki=0.1,
        kd=0.2
    )
    pid = PIDController(config)
    # ... existing code ... 