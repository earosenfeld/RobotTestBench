from robot_testbench.motor import MotorSimulator, MotorParameters

def test_motor_step_basic():
    """Test basic motor step response."""
    params = MotorParameters(
        inertia=0.05,
        damping=1.0,
        torque_constant=0.1,
        max_torque=4.0,
        max_speed=20.0,
        resistance=1.0,
        inductance=0.1
    )
    motor = MotorSimulator(params)
    dt = 0.01
    voltage = 5.0
    for _ in range(100):
        motor.step(dt, voltage)
    # ... existing code ...

def test_motor_step_zero_voltage():
    """Test motor step with zero voltage."""
    params = MotorParameters(
        inertia=0.05,
        damping=1.0,
        torque_constant=0.1,
        max_torque=4.0,
        max_speed=20.0,
        resistance=1.0,
        inductance=0.1
    )
    motor = MotorSimulator(params)
    dt = 0.01
    voltage = 0.0
    for _ in range(100):
        motor.step(dt, voltage)
    # ... existing code ...

def test_motor_step_negative_voltage():
    """Test motor step with negative voltage."""
    params = MotorParameters(
        inertia=0.05,
        damping=1.0,
        torque_constant=0.1,
        max_torque=4.0,
        max_speed=20.0,
        resistance=1.0,
        inductance=0.1
    )
    motor = MotorSimulator(params)
    dt = 0.01
    voltage = -5.0
    for _ in range(100):
        motor.step(dt, voltage)
    # ... existing code ... 