"""
Motor dashboard layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple

from ..visualization.plots import TimeSeriesPlot, XYPlot
from ..visualization.gauges import CircularGauge, LinearGauge

class MotorDashboard:
    """Motor dashboard layout."""
    
    def __init__(self):
        # Create plots
        self.position_plot = TimeSeriesPlot(
            "Position Control",
            "Time (s)",
            "Position (rad)"
        )
        self.velocity_plot = TimeSeriesPlot(
            "Velocity Control",
            "Time (s)",
            "Velocity (rad/s)"
        )
        self.torque_plot = TimeSeriesPlot(
            "Torque Control",
            "Time (s)",
            "Torque (N⋅m)"
        )
        self.phase_plot = XYPlot(
            "Phase Plot",
            "Position (rad)",
            "Velocity (rad/s)"
        )
        
        # Create gauges
        self.position_gauge = CircularGauge(
            "Position",
            -np.pi,
            np.pi
        )
        self.velocity_gauge = CircularGauge(
            "Velocity",
            -1000,
            1000
        )
        self.torque_gauge = LinearGauge(
            "Torque",
            -10,
            10
        )
        
        # Add lines to plots
        self.position_plot.add_line("Setpoint", color='r', style='--')
        self.position_plot.add_line("Actual", color='b')
        
        self.velocity_plot.add_line("Setpoint", color='r', style='--')
        self.velocity_plot.add_line("Actual", color='b')
        
        self.torque_plot.add_line("Command", color='r')
        self.torque_plot.add_line("Actual", color='b')
        
        self.phase_plot.add_line("Trajectory", color='b')
        
    def update(self, time: np.ndarray, data: Dict[str, np.ndarray]):
        """Update all dashboard components with new data."""
        # Update plots
        self.position_plot.update(time, {
            "Setpoint": data["position_setpoint"],
            "Actual": data["position"]
        })
        
        self.velocity_plot.update(time, {
            "Setpoint": data["velocity_setpoint"],
            "Actual": data["velocity"]
        })
        
        self.torque_plot.update(time, {
            "Command": data["torque_command"],
            "Actual": data["torque"]
        })
        
        self.phase_plot.update(
            data["position"],
            {"Trajectory": data["velocity"]}
        )
        
        # Update gauges
        self.position_gauge.update(data["position"][-1])
        self.velocity_gauge.update(data["velocity"][-1])
        self.torque_gauge.update(data["torque"][-1])
        
    def save(self, directory: str):
        """Save all dashboard components to files."""
        self.position_plot.save(f"{directory}/position.png")
        self.velocity_plot.save(f"{directory}/velocity.png")
        self.torque_plot.save(f"{directory}/torque.png")
        self.phase_plot.save(f"{directory}/phase.png")
        
        self.position_gauge.save(f"{directory}/position_gauge.png")
        self.velocity_gauge.save(f"{directory}/velocity_gauge.png")
        self.torque_gauge.save(f"{directory}/torque_gauge.png")
        
    def close(self):
        """Close all dashboard components."""
        self.position_plot.close()
        self.velocity_plot.close()
        self.torque_plot.close()
        self.phase_plot.close()
        
        self.position_gauge.close()
        self.velocity_gauge.close()
        self.torque_gauge.close() 