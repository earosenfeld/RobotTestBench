"""
Test plan execution engine for motor testing.
"""

import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
from robot_testbench.motor import MotorSimulator, MotorParameters
from robot_testbench.motor import LoadMotor, LoadMotorParameters, TestLoad

@dataclass
class TestStep:
    """Configuration for a single test step."""
    name: str
    duration: float  # seconds
    setpoint: float  # rad/s or rad
    load_type: str = 'constant_torque'
    load_params: dict = None
    control_mode: str = 'velocity'
    tolerance: float = 0.1  # rad/s or rad
    rise_time: float = None
    max_torque: float = None
    timeout: float = None

@dataclass
class TestPlan:
    """Complete test plan configuration."""
    name: str
    description: str
    motor_params: MotorParameters
    load_params: LoadMotorParameters
    steps: List[TestStep]
    sample_rate: float = 1000.0  # Hz
    data_log_path: Optional[str] = None

class TestExecutor:
    """Executes test plans and collects results."""
    
    def __init__(self, plan: TestPlan):
        """Initialize test executor with a test plan."""
        self.plan = plan
        self.motor = MotorSimulator(plan.motor_params)
        self.load = LoadMotor(plan.load_params)
        self.test_load = TestLoad()
        self.dt = 1.0 / plan.sample_rate
        self.results = []
        
    def _load_step(self, step: TestStep):
        """Configure load for a test step."""
        self.test_load.set_load_type(step.load_type, **step.load_params)
        
    def _check_specs(self, step: TestStep, data: Dict) -> Dict:
        """Check if test data meets specifications."""
        specs = {
            'passed': True,
            'errors': []
        }
        
        # Check setpoint tracking
        if step.control_mode == 'velocity':
            errors = [abs(v - step.setpoint) for v in data['velocity']]
            max_error = max(errors)
        else:
            errors = [abs(p - step.setpoint) for p in data['position']]
            max_error = max(errors)
        if max_error > step.tolerance:
            specs['passed'] = False
            specs['errors'].append(f"Setpoint tracking error: {max_error:.3f} > {step.tolerance}")
        
        # Check rise time
        if step.rise_time is not None:
            # Find time to reach 90% of setpoint
            target = 0.9 * step.setpoint
            if step.control_mode == 'velocity':
                rise_time = next((t for t, v in zip(data['time'], data['velocity']) 
                                if abs(v) >= abs(target)), None)
            else:
                rise_time = next((t for t, p in zip(data['time'], data['position']) 
                                if abs(p) >= abs(target)), None)
                
            if rise_time is None or rise_time > step.rise_time:
                specs['passed'] = False
                if rise_time is None:
                    specs['errors'].append(f"Rise time: N/A > {step.rise_time}s")
                else:
                    specs['errors'].append(f"Rise time: {rise_time:.3f}s > {step.rise_time}s")
                
        # Check max torque
        if step.max_torque is not None:
            max_torque = max(abs(t) for t in data['torque'])
            if max_torque > step.max_torque:
                specs['passed'] = False
                specs['errors'].append(f"Max torque: {max_torque:.3f} > {step.max_torque}")
                
        return specs
        
    def run_step(self, step: TestStep) -> Dict:
        """Run a single test step and return results."""
        self._load_step(step)
        
        # Initialize data collection
        data = {
            'time': [],
            'position': [],
            'velocity': [],
            'torque': [],
            'current': [],
            'temperature': [],
            'efficiency': []
        }
        
        # Run simulation
        for _ in range(int(step.duration * self.plan.sample_rate)):
            # Step motor
            pos, vel, curr = self.motor.step(self.dt, step.setpoint)
            # Step load
            load_pos, load_vel, load_torque = self.load.step(
                self.dt,
                pos,
                vel
            )
            # Collect data
            data['time'].append(self.motor._last_update_time)
            data['position'].append(pos)
            data['velocity'].append(vel)
            data['torque'].append(load_torque)
            data['current'].append(curr)
            data['temperature'].append(self.motor.temperature)
            # For efficiency, use get_efficiency if available
            if hasattr(self.motor, 'get_efficiency'):
                data['efficiency'].append(self.motor.get_efficiency())
            else:
                data['efficiency'].append(1.0)
        # Check specifications
        specs = self._check_specs(step, data)
        
        return {
            'step': step,
            'data': data,
            'specs': specs
        }
        
    def run(self) -> List[Dict]:
        """Run the complete test plan."""
        self.results = []
        for step in self.plan.steps:
            result = self.run_step(step)
            self.results.append(result)
            
            # Log results if path specified
            if self.plan.data_log_path:
                self._log_results(result)
                
        return self.results
    
    def _log_results(self, result: Dict):
        """Log test results to file."""
        log_path = Path(self.plan.data_log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Save data as CSV
        data = result['data']
        filename = f"step_{len(self.results)}.csv"
        with open(log_path / filename, 'w') as f:
            f.write("time,position,velocity,torque,current,temperature,efficiency\n")
            for i in range(len(data['time'])):
                f.write(f"{data['time'][i]:.6f},{data['position'][i]:.6f},"
                       f"{data['velocity'][i]:.6f},{data['torque'][i]:.6f},"
                       f"{data['current'][i]:.6f},{data['temperature'][i]:.6f},"
                       f"{data['efficiency'][i]:.6f}\n")
        
        # Save specs as YAML
        specs_filename = f"step_{len(self.results)}_specs.yaml"
        with open(log_path / specs_filename, 'w') as f:
            yaml.dump(result['specs'], f)

def load_test_plan(yaml_path: str) -> TestPlan:
    """Load a test plan from a YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Convert motor parameters
    motor_params = MotorParameters(**config['motor_params'])
    
    # Convert load parameters
    load_params = LoadMotorParameters(**config['load_params'])
    
    # Convert test steps
    steps = [TestStep(**step) for step in config['steps']]
    
    return TestPlan(
        name=config['name'],
        description=config['description'],
        motor_params=motor_params,
        load_params=load_params,
        steps=steps,
        sample_rate=config.get('sample_rate', 1000.0),
        data_log_path=config.get('data_log_path')
    ) 