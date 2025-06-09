"""
Sensor simulation module for RobotTestBench.
Implements noisy sensor signals and filtering options.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from scipy import signal

@dataclass
class SensorParameters:
    """Parameters for sensor simulation."""
    position_noise_std: float = 0.001  # rad
    velocity_noise_std: float = 0.01   # rad/s
    current_noise_std: float = 0.1     # A
    sampling_rate: float = 1000.0      # Hz
    filter_type: str = 'lowpass'       # 'lowpass' or 'kalman'
    filter_cutoff: float = 50.0        # Hz (for lowpass)
    kalman_process_noise: float = 0.01  # Reduced process noise
    kalman_measurement_noise: float = 0.1  # Reduced measurement noise

class SensorSimulator:
    """Simulates noisy sensor signals with filtering options."""
    
    def __init__(self, params: SensorParameters):
        """Initialize sensor simulator with parameters."""
        self.params = params
        self._setup_filters()
        
    def _setup_filters(self):
        """Set up the chosen filter."""
        if self.params.filter_type == 'lowpass':
            # Calculate window size based on cutoff frequency
            # Use a smaller window for better tracking
            self.window_size = int(self.params.sampling_rate / (4 * self.params.filter_cutoff))
            self.position_window = []
            self.velocity_window = []
            self.current_window = []
        elif self.params.filter_type == 'kalman':
            # Initialize Kalman filter state
            self.kalman_state = np.zeros(2)  # [position, velocity]
            self.kalman_cov = np.eye(2)  # Initial covariance
            self.dt = 1.0 / self.params.sampling_rate
            
    def _apply_lowpass_filter(self, x: np.ndarray, window: List[float]) -> np.ndarray:
        """Apply lowpass filter to signal x (batch or streaming)."""
        # If input is a single value, treat as streaming and maintain state
        if x.size == 1:
            # Add new value to window
            if len(window) >= self.window_size:
                window.pop(0)
            window.append(x[0])
            
            # Return mean of window
            return np.array([np.mean(window)])
        else:
            # For batch data, use moving average
            filtered = np.zeros_like(x)
            for i in range(len(x)):
                start = max(0, i - self.window_size + 1)
                filtered[i] = np.mean(x[start:i+1])
            return filtered
    
    def _apply_kalman_filter(self, measurement: np.ndarray) -> np.ndarray:
        """Apply Kalman filter to measurement."""
        # Prediction step
        F = np.array([[1, self.dt],
                     [0, 1]])
        self.kalman_state = F @ self.kalman_state
        self.kalman_cov = F @ self.kalman_cov @ F.T + np.eye(2) * self.params.kalman_process_noise
        
        # Update step
        H = np.array([[1, 0]])  # Measurement matrix
        K = self.kalman_cov @ H.T @ np.linalg.inv(H @ self.kalman_cov @ H.T + self.params.kalman_measurement_noise)
        self.kalman_state = self.kalman_state + K @ (measurement - H @ self.kalman_state)
        self.kalman_cov = (np.eye(2) - K @ H) @ self.kalman_cov
        
        return self.kalman_state
    
    def add_noise(self, position: float, velocity: float, current: float) -> Tuple[float, float, float]:
        """Add noise to sensor signals."""
        noisy_position = position + np.random.normal(0, self.params.position_noise_std)
        noisy_velocity = velocity + np.random.normal(0, self.params.velocity_noise_std)
        noisy_current = current + np.random.normal(0, self.params.current_noise_std)
        return noisy_position, noisy_velocity, noisy_current
    
    def filter_signals(self, position: float, velocity: float, current: float) -> Tuple[float, float, float]:
        """Apply filtering to sensor signals."""
        if self.params.filter_type == 'lowpass':
            # Convert to arrays for filtering
            pos_array = np.array([position])
            vel_array = np.array([velocity])
            curr_array = np.array([current])
            
            # Apply filters with separate windows
            filtered_position = self._apply_lowpass_filter(pos_array, self.position_window)[0]
            filtered_velocity = self._apply_lowpass_filter(vel_array, self.velocity_window)[0]
            filtered_current = self._apply_lowpass_filter(curr_array, self.current_window)[0]
            
        elif self.params.filter_type == 'kalman':
            # For Kalman filter, we use position and velocity together
            measurement = np.array([position])
            filtered_state = self._apply_kalman_filter(measurement)
            filtered_position = filtered_state[0]
            filtered_velocity = filtered_state[1]
            filtered_current = current  # Kalman filter not applied to current
            
        return filtered_position, filtered_velocity, filtered_current
    
    def process_signals(self, position: float, velocity: float, current: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Process sensor signals: add noise and apply filtering.
        
        Returns:
            Tuple of (raw_signals, filtered_signals)
            where each is a tuple of (position, velocity, current)
        """
        # Add noise
        raw_position, raw_velocity, raw_current = self.add_noise(position, velocity, current)
        
        # Apply filtering
        filtered_position, filtered_velocity, filtered_current = self.filter_signals(
            raw_position, raw_velocity, raw_current
        )
        
        return (
            (raw_position, raw_velocity, raw_current),
            (filtered_position, filtered_velocity, filtered_current)
        ) 