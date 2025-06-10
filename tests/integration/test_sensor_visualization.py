import numpy as np
import matplotlib.pyplot as plt
from robot_testbench.sensors.sensors import SensorSimulator, SensorParameters

def test_sensor_visualization():
    """Test and visualize sensor signal processing."""
    # Create sensor parameters
    sensor_params = SensorParameters(
        position_noise_std=0.001,
        velocity_noise_std=0.01,
        current_noise_std=0.1,
        sampling_rate=1000.0,
        filter_type='lowpass',
        filter_cutoff=50.0,
        kalman_process_noise=0.1,
        kalman_measurement_noise=1.0
    )
    
    # Create sensor simulator
    sensor = SensorSimulator(sensor_params)
    
    # Generate test signals
    t = np.linspace(0, 1, 1000)  # 1 second of data at 1000 Hz
    position = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine wave
    velocity = 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * t)  # Derivative of position
    current = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    
    # Process signals
    raw_positions = []
    raw_velocities = []
    raw_currents = []
    filtered_positions = []
    filtered_velocities = []
    filtered_currents = []
    
    for i in range(len(t)):
        (raw_pos, raw_vel, raw_curr), (filt_pos, filt_vel, filt_curr) = sensor.process_signals(
            position[i], velocity[i], current[i]
        )
        raw_positions.append(raw_pos)
        raw_velocities.append(raw_vel)
        raw_currents.append(raw_curr)
        filtered_positions.append(filt_pos)
        filtered_velocities.append(filt_vel)
        filtered_currents.append(filt_curr)
    
    # Convert to numpy arrays
    raw_positions = np.array(raw_positions)
    raw_velocities = np.array(raw_velocities)
    raw_currents = np.array(raw_currents)
    filtered_positions = np.array(filtered_positions)
    filtered_velocities = np.array(filtered_velocities)
    filtered_currents = np.array(filtered_currents)
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Sensor Signal Processing Visualization')
    
    # Position plot
    ax1.plot(t, position, 'k--', label='True Position', alpha=0.5)
    ax1.plot(t, raw_positions, 'r.', label='Raw Position', alpha=0.3)
    ax1.plot(t, filtered_positions, 'b-', label='Filtered Position')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True)
    
    # Velocity plot
    ax2.plot(t, velocity, 'k--', label='True Velocity', alpha=0.5)
    ax2.plot(t, raw_velocities, 'r.', label='Raw Velocity', alpha=0.3)
    ax2.plot(t, filtered_velocities, 'b-', label='Filtered Velocity')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True)
    
    # Current plot
    ax3.plot(t, current, 'k--', label='True Current', alpha=0.5)
    ax3.plot(t, raw_currents, 'r.', label='Raw Current', alpha=0.3)
    ax3.plot(t, filtered_currents, 'b-', label='Filtered Current')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('sensor_visualization.png')
    plt.close()
    
    # Print statistics
    print("\nSignal Statistics:")
    print("Position:")
    print(f"  Raw std: {np.std(raw_positions):.6f}")
    print(f"  Filtered std: {np.std(filtered_positions):.6f}")
    print(f"  Raw mean error: {np.abs(np.mean(raw_positions) - np.mean(position)):.6f}")
    print(f"  Filtered mean error: {np.abs(np.mean(filtered_positions) - np.mean(position)):.6f}")
    
    print("\nVelocity:")
    print(f"  Raw std: {np.std(raw_velocities):.6f}")
    print(f"  Filtered std: {np.std(filtered_velocities):.6f}")
    print(f"  Raw mean error: {np.abs(np.mean(raw_velocities) - np.mean(velocity)):.6f}")
    print(f"  Filtered mean error: {np.abs(np.mean(filtered_velocities) - np.mean(velocity)):.6f}")
    
    print("\nCurrent:")
    print(f"  Raw std: {np.std(raw_currents):.6f}")
    print(f"  Filtered std: {np.std(filtered_currents):.6f}")
    print(f"  Raw mean error: {np.abs(np.mean(raw_currents) - np.mean(current)):.6f}")
    print(f"  Filtered mean error: {np.abs(np.mean(filtered_currents) - np.mean(current)):.6f}")

if __name__ == '__main__':
    test_sensor_visualization() 