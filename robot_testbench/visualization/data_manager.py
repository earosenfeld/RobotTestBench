"""
Data management for the RobotTestBench dashboard.
"""

import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

class TestDataManager:
    def __init__(self):
        self.current_test_data: List[Dict[str, Any]] = []
        self.test_start_time: datetime = None
        self.is_test_running: bool = False
        
    def start_test(self):
        """Start a new test."""
        self.current_test_data = []
        self.test_start_time = datetime.now()
        self.is_test_running = True
        
    def stop_test(self):
        """Stop the current test."""
        self.is_test_running = False
        
    def add_data_point(self, position: float, velocity: float, torque: float):
        """Add a new data point to the current test."""
        if not self.is_test_running:
            return
            
        elapsed_time = (datetime.now() - self.test_start_time).total_seconds()
        self.current_test_data.append({
            'elapsed_time': elapsed_time,
            'position': position,
            'velocity': velocity,
            'torque': torque
        })
        
    def get_current_data(self) -> str:
        """Get the current test data as a JSON string."""
        return json.dumps(self.current_test_data)
        
    def get_dataframe(self) -> pd.DataFrame:
        """Get the current test data as a pandas DataFrame."""
        return pd.DataFrame(self.current_test_data)
        
    def save_test_data(self, filename: str):
        """Save the current test data to a file."""
        if not self.current_test_data:
            return
            
        df = self.get_dataframe()
        df.to_csv(filename, index=False) 