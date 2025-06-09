"""
Data acquisition and logging module for RobotTestBench.
Handles real-time data streaming and storage.
"""

import json
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

@dataclass
class MotorTestMetadata:
    """Metadata for motor test data."""
    test_name: str
    timestamp: str
    motor_params: dict
    pid_gains: dict
    control_mode: str
    setpoint: float
    duration: float
    sample_rate: float
    sensor_params: Optional[dict] = None  # Added sensor parameters

class DataLogger:
    """Handles data acquisition and logging for test runs."""
    
    def __init__(self, data_dir: str = "data/logs"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_test: Optional[MotorTestMetadata] = None
        self.csv_file: Optional[csv.Writer] = None
        self.csv_handle = None
        self.start_time: Optional[float] = None
        
    def start_test(self, metadata: MotorTestMetadata):
        """Start a new test run."""
        if self.current_test is not None:
            self.end_test()
            
        self.current_test = metadata
        self.start_time = time.time()
        
        # Create test directory
        test_dir = self.data_dir / metadata.test_name
        test_dir.mkdir(exist_ok=True)
        
        # Save metadata
        with open(test_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)
            
        # Initialize CSV file
        self.csv_handle = open(test_dir / "data.csv", "w", newline="")
        self.csv_file = csv.writer(self.csv_handle)
        
        # Write header with both raw and filtered data
        self.csv_file.writerow([
            "timestamp",
            "elapsed_time",
            "raw_position",
            "raw_velocity",
            "raw_current",
            "filtered_position",
            "filtered_velocity",
            "filtered_current",
            "torque",
            "control_output",
            "error"
        ])
        
    def log_data(self, data: Dict[str, float]):
        """Log a data point with both raw and filtered sensor data."""
        if self.current_test is None or self.csv_file is None:
            raise RuntimeError("No active test run")
            
        elapsed = time.time() - self.start_time
        self.csv_file.writerow([
            datetime.now().isoformat(),
            elapsed,
            data.get("raw_position", 0.0),
            data.get("raw_velocity", 0.0),
            data.get("raw_current", 0.0),
            data.get("filtered_position", 0.0),
            data.get("filtered_velocity", 0.0),
            data.get("filtered_current", 0.0),
            data.get("torque", 0.0),
            data.get("control_output", 0.0),
            data.get("error", 0.0)
        ])
        
    def end_test(self):
        """End the current test run."""
        if self.csv_handle is not None:
            self.csv_handle.close()
            self.csv_handle = None
        self.csv_file = None
        self.current_test = None
        self.start_time = None
        
    def get_test_data(self, test_name: str) -> Dict[str, Any]:
        """Load test data from file."""
        test_dir = self.data_dir / test_name
        
        # Load metadata
        with open(test_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        # Load data
        data = []
        with open(test_dir / "data.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
                
        return {
            "metadata": metadata,
            "data": data
        } 