import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class QuadratureEncoderConfig:
    """Configuration for quadrature encoder simulation."""
    resolution: int  # Counts per revolution
    noise_std: float = 0.0  # Standard deviation of noise in counts
    quantization_error: float = 0.0  # Maximum quantization error in counts
    edge_trigger_noise: float = 0.0  # Standard deviation of edge trigger timing noise in seconds
    max_frequency: float = 1000.0  # Maximum frequency in Hz
    calibration_offset: float = 0.0  # Calibration offset in radians

class QuadratureEncoder:
    """Simulates a quadrature encoder with realistic characteristics."""
    
    def __init__(self, config: QuadratureEncoderConfig):
        self.config = config
        self.last_position = 0.0
        self.last_time = 0.0
        self.counts = 0
        self._reset_internal_state()
    
    def _reset_internal_state(self):
        """Reset internal state variables."""
        self.last_position = 0.0
        self.last_time = 0.0
        self.counts = 0
    
    def _quantize_position(self, position: float) -> int:
        """Convert continuous position to quantized counts."""
        counts = int(position * self.config.resolution / (2 * np.pi))
        return counts
    
    def _add_noise(self, counts: int) -> int:
        """Add noise to the encoder counts."""
        if self.config.noise_std > 0:
            noise = np.random.normal(0, self.config.noise_std)
            counts += int(round(noise))
        return counts
    
    def _simulate_edge_trigger(self, dt: float) -> float:
        """Simulate edge trigger timing noise."""
        if self.config.edge_trigger_noise > 0:
            return np.random.normal(dt, self.config.edge_trigger_noise)
        return dt
    
    def update(self, position: float, velocity: float, dt: float) -> Tuple[int, int]:
        """Update encoder state and return A/B channel counts."""
        # Check frequency limit
        if abs(velocity) * self.config.resolution / (2 * np.pi) > self.config.max_frequency:
            raise ValueError("Encoder frequency limit exceeded")
        
        # Quantize position
        current_counts = self._quantize_position(position + self.config.calibration_offset)
        
        # Add noise
        noisy_counts = self._add_noise(current_counts)
        
        # Calculate count difference
        count_diff = noisy_counts - self.counts
        
        # Simulate edge trigger timing
        effective_dt = self._simulate_edge_trigger(dt)
        
        # Update internal state
        self.counts = noisy_counts
        self.last_position = position
        self.last_time += effective_dt
        
        # Generate quadrature signals (A and B channels)
        # A leads B for positive rotation, B leads A for negative rotation
        a_channel = (self.counts % 4) // 2
        b_channel = (self.counts % 4) % 2
        
        return a_channel, b_channel

@dataclass
class ForceTorqueSensorConfig:
    """Configuration for force/torque sensor simulation."""
    sensitivity: float  # N⋅m/V or N/V
    noise_std: float = 0.0  # Standard deviation of noise in V
    drift_rate: float = 0.0  # Drift rate in V/s
    hysteresis: float = 0.0  # Hysteresis in N⋅m or N
    calibration_offset: float = 0.0  # Calibration offset in V
    temperature_coefficient: float = 0.0  # Temperature coefficient in V/°C
    max_range: float = float('inf')  # Maximum measurable force/torque

class ForceTorqueSensor:
    """Simulates a force/torque sensor with realistic characteristics."""
    
    def __init__(self, config: ForceTorqueSensorConfig):
        self.config = config
        self._reset_internal_state()
    
    def _reset_internal_state(self):
        """Reset internal state variables."""
        self.last_reading = 0.0
        self.drift_offset = 0.0
        self.temperature = 25.0  # Default temperature in °C
        self.history = []  # For hysteresis calculation
    
    def set_temperature(self, temperature: float):
        """Set the sensor temperature."""
        self.temperature = temperature
    
    def _calculate_hysteresis(self, current_value: float) -> float:
        """Calculate hysteresis effect based on measurement history."""
        if not self.history:
            self.history.append(current_value)
            return 0.0
        
        # Simple hysteresis model
        last_value = self.history[-1]
        direction = np.sign(current_value - last_value)
        hysteresis = direction * self.config.hysteresis
        
        # Update history (keep last 10 readings)
        self.history.append(current_value)
        if len(self.history) > 10:
            self.history.pop(0)
        
        return hysteresis
    
    def update(self, force_torque: float, dt: float) -> float:
        """Update sensor state and return voltage reading."""
        # Check range limit
        if abs(force_torque) > self.config.max_range:
            raise ValueError("Force/torque exceeds sensor range")
        
        # Calculate base reading
        base_reading = force_torque / self.config.sensitivity
        
        # Add calibration offset
        reading = base_reading + self.config.calibration_offset
        
        # Add temperature effect
        temp_effect = (self.temperature - 25.0) * self.config.temperature_coefficient
        reading += temp_effect
        
        # Add drift
        self.drift_offset += self.config.drift_rate * dt
        reading += self.drift_offset
        
        # Add hysteresis
        reading += self._calculate_hysteresis(reading)
        
        # Add noise
        if self.config.noise_std > 0:
            reading += np.random.normal(0, self.config.noise_std)
        
        self.last_reading = reading
        return reading

@dataclass
class JointAngleSensorConfig:
    """Configuration for joint angle sensor simulation."""
    resolution: float  # Radians per count
    noise_std: float = 0.0  # Standard deviation of noise in radians
    backlash: float = 0.0  # Backlash in radians
    limit_stops: Tuple[float, float] = (-np.pi, np.pi)  # Min and max angles
    calibration_offset: float = 0.0  # Calibration offset in radians
    temperature_coefficient: float = 0.0  # Temperature coefficient in rad/°C

class JointAngleSensor:
    """Simulates a joint angle sensor with realistic characteristics."""
    
    def __init__(self, config: JointAngleSensorConfig):
        self.config = config
        self._reset_internal_state()
    
    def _reset_internal_state(self):
        """Reset internal state variables."""
        self.last_angle = 0.0
        self.last_direction = 0
        self.temperature = 25.0  # Default temperature in °C
    
    def set_temperature(self, temperature: float):
        """Set the sensor temperature."""
        self.temperature = temperature
    
    def _apply_backlash(self, angle: float, velocity: float) -> float:
        """Apply backlash effect to the angle measurement."""
        if abs(velocity) < 1e-6:  # If velocity is very small
            return angle
        
        direction = np.sign(velocity)
        if direction != self.last_direction:
            # Direction change, apply backlash
            angle += direction * self.config.backlash
            self.last_direction = direction
        
        return angle
    
    def update(self, angle: float, velocity: float) -> float:
        """Update sensor state and return angle reading."""
        # Check limit stops
        if angle < self.config.limit_stops[0] or angle > self.config.limit_stops[1]:
            raise ValueError("Angle exceeds limit stops")
        
        # Apply backlash
        angle = self._apply_backlash(angle, velocity)
        
        # Add calibration offset
        reading = angle + self.config.calibration_offset
        
        # Add temperature effect
        temp_effect = (self.temperature - 25.0) * self.config.temperature_coefficient
        reading += temp_effect
        
        # Add noise
        if self.config.noise_std > 0:
            reading += np.random.normal(0, self.config.noise_std)
        
        # Quantize to resolution
        reading = round(reading / self.config.resolution) * self.config.resolution
        
        self.last_angle = reading
        return reading 