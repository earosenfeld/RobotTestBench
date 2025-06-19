"""
Test plan definition and execution.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml

@dataclass
class Step:
    """A single step in a test plan."""
    name: str
    duration: float
    setpoint: float
    load_type: str = 'constant_torque'
    load_params: dict = None
    control_mode: str = 'velocity'
    tolerance: float = 0.1
    rise_time: float = None
    max_torque: float = None
    timeout: float = None

@dataclass
class Plan:
    """A complete test plan with multiple steps."""
    name: str
    description: str
    steps: List[Step]
    metadata: Dict[str, str] = None

class Executor:
    """Executes a test plan and collects results."""
    
    def __init__(self, test_plan: Plan):
        self.test_plan = test_plan
        self.current_step = 0
        self.results = []
        
    def step(self, current_time: float, current_value: float) -> Optional[float]:
        """
        Execute one step of the test plan.
        
        Args:
            current_time: Current time in seconds
            current_value: Current value (e.g., position or velocity)
            
        Returns:
            Setpoint for the current step, or None if test is complete
        """
        if self.current_step >= len(self.test_plan.steps):
            return None
            
        step = self.test_plan.steps[self.current_step]
        
        # Check if step is complete
        if abs(current_value - step.setpoint) <= step.tolerance:
            self.current_step += 1
            if self.current_step >= len(self.test_plan.steps):
                return None
            return self.test_plan.steps[self.current_step].setpoint
            
        return step.setpoint

def load_test_plan(file_path: Union[str, Path]) -> Plan:
    """
    Load a test plan from a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Plan object
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        
    steps = []
    for step_data in data['steps']:
        step = Step(
            name=step_data['name'],
            duration=step_data['duration'],
            setpoint=step_data['setpoint'],
            load_type=step_data.get('load_type', 'constant_torque'),
            load_params=step_data.get('load_params', {}),
            control_mode=step_data.get('control_mode', 'velocity'),
            tolerance=step_data.get('tolerance', 0.1),
            rise_time=step_data.get('rise_time'),
            max_torque=step_data.get('max_torque'),
            timeout=step_data.get('timeout')
        )
        steps.append(step)
        
    return Plan(
        name=data['name'],
        description=data['description'],
        steps=steps,
        metadata=data.get('metadata', {})
    ) 