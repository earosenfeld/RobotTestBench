"""
Plot components for the dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.collections import PathCollection

class TimeSeriesPlot:
    """Time series plot component."""
    
    def __init__(self, title: str, xlabel: str, ylabel: str):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._lines: Dict[str, plt.Line2D] = {}
        
    def create(self) -> Tuple[Figure, Axes]:
        """Create the plot figure and axes."""
        self._fig, self._ax = plt.subplots(figsize=(10, 6))
        self._ax.set_title(self.title)
        self._ax.set_xlabel(self.xlabel)
        self._ax.set_ylabel(self.ylabel)
        self._ax.grid(True)
        return self._fig, self._ax
        
    def add_line(self, name: str, color: str = 'b', style: str = '-') -> plt.Line2D:
        """Add a new line to the plot."""
        if self._ax is None:
            self.create()
        line, = self._ax.plot([], [], color=color, linestyle=style, label=name)
        self._lines[name] = line
        self._ax.legend()
        return line
        
    def update(self, time: np.ndarray, data: Dict[str, np.ndarray]):
        """Update the plot with new data."""
        if self._ax is None:
            self.create()
            
        for name, values in data.items():
            if name in self._lines:
                self._lines[name].set_data(time, values)
                
        self._ax.relim()
        self._ax.autoscale_view()
        
    def save(self, filename: str):
        """Save the plot to a file."""
        if self._fig is not None:
            self._fig.savefig(filename)
            
    def close(self):
        """Close the plot."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

class XYPlot:
    """XY plot component."""
    
    def __init__(self, title: str, xlabel: str, ylabel: str):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._lines: Dict[str, plt.Line2D] = {}
        
    def create(self) -> Tuple[Figure, Axes]:
        """Create the plot figure and axes."""
        self._fig, self._ax = plt.subplots(figsize=(10, 6))
        self._ax.set_title(self.title)
        self._ax.set_xlabel(self.xlabel)
        self._ax.set_ylabel(self.ylabel)
        self._ax.grid(True)
        return self._fig, self._ax
        
    def add_line(self, name: str, color: str = 'b', style: str = '-') -> plt.Line2D:
        """Add a new line to the plot."""
        if self._ax is None:
            self.create()
        line, = self._ax.plot([], [], color=color, linestyle=style, label=name)
        self._lines[name] = line
        self._ax.legend()
        return line
        
    def update(self, x: np.ndarray, y: Dict[str, np.ndarray]):
        """Update the plot with new data."""
        if self._ax is None:
            self.create()
            
        for name, values in y.items():
            if name in self._lines:
                self._lines[name].set_data(x, values)
                
        self._ax.relim()
        self._ax.autoscale_view()
        
    def save(self, filename: str):
        """Save the plot to a file."""
        if self._fig is not None:
            self._fig.savefig(filename)
            
    def close(self):
        """Close the plot."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

class ScatterPlot:
    """Scatter plot component."""
    
    def __init__(self, title: str, xlabel: str, ylabel: str):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._scatters: Dict[str, plt.PathCollection] = {}
        
    def create(self) -> Tuple[Figure, Axes]:
        """Create the plot figure and axes."""
        self._fig, self._ax = plt.subplots(figsize=(10, 6))
        self._ax.set_title(self.title)
        self._ax.set_xlabel(self.xlabel)
        self._ax.set_ylabel(self.ylabel)
        self._ax.grid(True)
        return self._fig, self._ax
        
    def add_scatter(self, name: str, color: str = 'b', marker: str = 'o') -> PathCollection:
        """Add a new scatter plot."""
        if self._ax is None:
            self.create()
        scatter = self._ax.scatter([], [], color=color, marker=marker, label=name)
        self._scatters[name] = scatter
        self._ax.legend()
        return scatter
        
    def update(self, x: np.ndarray, y: Dict[str, np.ndarray]):
        """Update the plot with new data."""
        if self._ax is None:
            self.create()
            
        for name, values in y.items():
            if name in self._scatters:
                self._scatters[name].set_offsets(np.column_stack([x, values]))
                
        self._ax.relim()
        self._ax.autoscale_view()
        
    def save(self, filename: str):
        """Save the plot to a file."""
        if self._fig is not None:
            self._fig.savefig(filename)
            
    def close(self):
        """Close the plot."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None 