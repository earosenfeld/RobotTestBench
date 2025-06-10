"""
Gauge components for the dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple, Union

class Gauge:
    """Base gauge component."""
    
    def __init__(self, title: str, min_value: float, max_value: float):
        self.title = title
        self.min_value = min_value
        self.max_value = max_value
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        
    def create(self) -> Tuple[Figure, Axes]:
        """Create the gauge figure and axes."""
        self._fig, self._ax = plt.subplots(figsize=(6, 6))
        self._ax.set_title(self.title)
        self._ax.set_aspect('equal')
        self._ax.axis('off')
        return self._fig, self._ax
        
    def update(self, value: float):
        """Update the gauge with a new value."""
        raise NotImplementedError
        
    def save(self, filename: str):
        """Save the gauge to a file."""
        if self._fig is not None:
            self._fig.savefig(filename)
            
    def close(self):
        """Close the gauge."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

class CircularGauge(Gauge):
    """Circular gauge component."""
    
    def __init__(self, title: str, min_value: float, max_value: float, 
                 start_angle: float = 180, end_angle: float = 0):
        super().__init__(title, min_value, max_value)
        self.start_angle = start_angle
        self.end_angle = end_angle
        self._arc: Optional[plt.Patch] = None
        self._needle: Optional[plt.Line2D] = None
        self._value_text: Optional[plt.Text] = None
        
    def create(self) -> Tuple[Figure, Axes]:
        """Create the gauge figure and axes."""
        fig, ax = super().create()
        
        # Draw gauge arc
        arc = plt.matplotlib.patches.Arc(
            (0.5, 0.5), 1, 1,
            theta1=self.start_angle,
            theta2=self.end_angle,
            color='gray'
        )
        ax.add_patch(arc)
        
        # Add value text
        self._value_text = ax.text(
            0.5, 0.3,
            f"{self.min_value:.1f}",
            ha='center',
            va='center',
            fontsize=12
        )
        
        return fig, ax
        
    def update(self, value: float):
        """Update the gauge with a new value."""
        if self._ax is None:
            self.create()
            
        # Clamp value to range
        value = np.clip(value, self.min_value, self.max_value)
        
        # Calculate needle angle
        angle_range = self.end_angle - self.start_angle
        value_range = self.max_value - self.min_value
        angle = self.start_angle + (value - self.min_value) / value_range * angle_range
        
        # Update needle
        if self._needle is not None:
            self._needle.remove()
            
        self._needle = self._ax.plot(
            [0.5, 0.5 + 0.4 * np.cos(np.radians(angle))],
            [0.5, 0.5 + 0.4 * np.sin(np.radians(angle))],
            color='red',
            linewidth=2
        )[0]
        
        # Update value text
        if self._value_text is not None:
            self._value_text.set_text(f"{value:.1f}")

class LinearGauge(Gauge):
    """Linear gauge component."""
    
    def __init__(self, title: str, min_value: float, max_value: float,
                 orientation: str = 'horizontal'):
        super().__init__(title, min_value, max_value)
        self.orientation = orientation
        self._bar: Optional[plt.Rectangle] = None
        self._value_text: Optional[plt.Text] = None
        
    def create(self) -> Tuple[Figure, Axes]:
        """Create the gauge figure and axes."""
        fig, ax = super().create()
        
        # Draw gauge background
        if self.orientation == 'horizontal':
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.2)
            self._bar = plt.Rectangle((0, 0), 0, 0.2, color='gray')
        else:
            ax.set_xlim(0, 0.2)
            ax.set_ylim(0, 1)
            self._bar = plt.Rectangle((0, 0), 0.2, 0, color='gray')
            
        ax.add_patch(self._bar)
        
        # Add value text
        self._value_text = ax.text(
            0.5, 0.5,
            f"{self.min_value:.1f}",
            ha='center',
            va='center',
            fontsize=12
        )
        
        return fig, ax
        
    def update(self, value: float):
        """Update the gauge with a new value."""
        if self._ax is None:
            self.create()
            
        # Clamp value to range
        value = np.clip(value, self.min_value, self.max_value)
        
        # Calculate bar size
        value_range = self.max_value - self.min_value
        size = (value - self.min_value) / value_range
        
        # Update bar
        if self._bar is not None:
            if self.orientation == 'horizontal':
                self._bar.set_width(size)
            else:
                self._bar.set_height(size)
                
        # Update value text
        if self._value_text is not None:
            self._value_text.set_text(f"{value:.1f}") 