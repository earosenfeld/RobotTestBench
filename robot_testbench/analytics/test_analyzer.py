"""
Test analysis module for RobotTestBench.
Computes performance metrics and generates reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Performance metrics for a test run."""
    rise_time: float  # Time to reach 90% of setpoint
    settling_time: float  # Time to settle within 5% of setpoint
    overshoot: float  # Maximum overshoot percentage
    steady_state_error: float  # Final error
    rms_error: float  # Root mean square error
    max_error: float  # Maximum error
    peak_torque: float  # Maximum torque
    peak_velocity: float  # Maximum velocity

class Analyzer:
    """Analyzes test data and computes performance metrics."""
    
    def __init__(self, data: pd.DataFrame, setpoint: float):
        self.data = data
        self.setpoint = setpoint
        self.metrics: Optional[PerformanceMetrics] = None
        
    def compute_metrics(self) -> PerformanceMetrics:
        """Compute performance metrics for the test run."""
        # Convert data to numpy arrays for faster computation
        time = self.data['elapsed_time'].values
        position = self.data['filtered_position'].values
        velocity = self.data['filtered_velocity'].values
        torque = self.data['torque'].values
        
        # Compute error
        error = self.setpoint - position
        
        # Rise time (time to reach 90% of setpoint)
        target = 0.9 * self.setpoint
        rise_time_idx = np.where(position >= target)[0]
        rise_time = time[rise_time_idx[0]] if len(rise_time_idx) > 0 else np.inf
        
        # Settling time (time to settle within 5% of setpoint)
        settled = np.abs(error) <= 0.05 * np.abs(self.setpoint)
        settling_time = time[np.where(settled)[0][0]] if np.any(settled) else np.inf
        
        # Overshoot
        if self.setpoint > 0:
            overshoot = (np.max(position) - self.setpoint) / self.setpoint * 100
        else:
            overshoot = (np.min(position) - self.setpoint) / self.setpoint * 100
            
        # Steady state error (average of last 10% of data)
        steady_state_error = np.mean(error[-len(error)//10:])
        
        # RMS error
        rms_error = np.sqrt(np.mean(error**2))
        
        # Maximum error
        max_error = np.max(np.abs(error))
        
        # Peak values
        peak_torque = np.max(np.abs(torque))
        peak_velocity = np.max(np.abs(velocity))
        
        self.metrics = PerformanceMetrics(
            rise_time=rise_time,
            settling_time=settling_time,
            overshoot=overshoot,
            steady_state_error=steady_state_error,
            rms_error=rms_error,
            max_error=max_error,
            peak_torque=peak_torque,
            peak_velocity=peak_velocity
        )
        
        return self.metrics
    
    def generate_plots(self, save_dir: Path):
        """Generate and save analysis plots."""
        if self.metrics is None:
            self.compute_metrics()
            
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Position plot
        ax1.plot(self.data['elapsed_time'], self.data['filtered_position'], label='Position')
        ax1.axhline(y=self.setpoint, color='r', linestyle='--', label='Setpoint')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title('Position Response')
        ax1.grid(True)
        ax1.legend()
        
        # Velocity plot
        ax2.plot(self.data['elapsed_time'], self.data['filtered_velocity'], label='Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (rad/s)')
        ax2.set_title('Velocity Profile')
        ax2.grid(True)
        ax2.legend()
        
        # Torque plot
        ax3.plot(self.data['elapsed_time'], self.data['torque'], label='Torque')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Torque (N⋅m)')
        ax3.set_title('Torque Profile')
        ax3.grid(True)
        ax3.legend()
        
        # Add metrics text
        metrics_text = (
            f"Rise Time: {self.metrics.rise_time:.3f} s\n"
            f"Settling Time: {self.metrics.settling_time:.3f} s\n"
            f"Overshoot: {self.metrics.overshoot:.1f}%\n"
            f"Steady State Error: {self.metrics.steady_state_error:.3f} rad\n"
            f"RMS Error: {self.metrics.rms_error:.3f} rad\n"
            f"Max Error: {self.metrics.max_error:.3f} rad\n"
            f"Peak Torque: {self.metrics.peak_torque:.2f} N⋅m\n"
            f"Peak Velocity: {self.metrics.peak_velocity:.2f} rad/s"
        )
        fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace')
        
        # Save plots
        plt.tight_layout()
        plt.savefig(save_dir / 'analysis_plots.png')
        plt.close()
        
    def generate_report(self, save_dir: Path):
        """Generate a comprehensive test report."""
        if self.metrics is None:
            self.compute_metrics()
            
        # Generate plots
        self.generate_plots(save_dir)
        
        # Create report text
        report = f"""Test Analysis Report
===================

Performance Metrics:
------------------
Rise Time: {self.metrics.rise_time:.3f} s
Settling Time: {self.metrics.settling_time:.3f} s
Overshoot: {self.metrics.overshoot:.1f}%
Steady State Error: {self.metrics.steady_state_error:.3f} rad
RMS Error: {self.metrics.rms_error:.3f} rad
Max Error: {self.metrics.max_error:.3f} rad
Peak Torque: {self.metrics.peak_torque:.2f} N⋅m
Peak Velocity: {self.metrics.peak_velocity:.2f} rad/s

Analysis:
--------
"""
        
        # Add analysis comments
        if self.metrics.overshoot > 20:
            report += "- High overshoot indicates aggressive control gains\n"
        if self.metrics.steady_state_error > 0.1:
            report += "- Significant steady state error suggests need for integral gain adjustment\n"
        if self.metrics.rise_time > 1.0:
            report += "- Slow rise time indicates conservative control gains\n"
            
        # Save report
        with open(save_dir / 'analysis_report.txt', 'w') as f:
            f.write(report) 