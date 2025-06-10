import numpy as np
import pandas as pd
from pathlib import Path

def create_harmonic_drive_data():
    # Parameters (should match metadata.json)
    duration = 10.0
    sample_rate = 1000.0
    dt = 1.0 / sample_rate
    n_samples = int(duration * sample_rate)
    time = np.arange(0, duration, dt)
    setpoint = 1.5
    wn = 1.2  # natural frequency (slower than small motors)
    zeta = 0.7  # higher damping for harmonic drive
    # Simulate position step response
    position = setpoint * (1 - np.exp(-zeta * wn * time) * (
        np.cos(wn * np.sqrt(1 - zeta**2) * time) +
        zeta / np.sqrt(1 - zeta**2) * np.sin(wn * np.sqrt(1 - zeta**2) * time)
    ))
    # Velocity and acceleration
    velocity = np.gradient(position, dt)
    acceleration = np.gradient(velocity, dt)
    # Torque (inertia*accel + damping*vel, scaled by gear ratio)
    inertia = 0.08
    damping = 0.15
    gear_ratio = 100.0
    torque = (inertia * acceleration + damping * velocity) * gear_ratio
    # Current (torque / torque constant)
    torque_constant = 0.12
    current = torque / torque_constant
    # Add noise
    position_noise = np.random.normal(0, 0.002, n_samples)
    velocity_noise = np.random.normal(0, 0.02, n_samples)
    current_noise = np.random.normal(0, 0.01, n_samples)
    # Raw and filtered signals
    raw_position = position + position_noise
    filtered_position = position
    raw_velocity = velocity + velocity_noise
    filtered_velocity = velocity
    raw_current = current + current_noise
    filtered_current = current
    # Torque sensor (add small noise)
    torque_noise = np.random.normal(0, 0.05, n_samples)
    torque_sensor = torque + torque_noise
    # Build DataFrame
    df = pd.DataFrame({
        'timestamp': time,
        'elapsed_time': time,
        'raw_position': raw_position,
        'filtered_position': filtered_position,
        'raw_velocity': raw_velocity,
        'filtered_velocity': filtered_velocity,
        'raw_current': raw_current,
        'filtered_current': filtered_current,
        'torque': torque_sensor
    })
    # Save to CSV
    out_path = Path('data/logs/HarmonicDrive_HumanoidJoint/data.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Sample Harmonic Drive data saved to {out_path}")

if __name__ == "__main__":
    create_harmonic_drive_data() 