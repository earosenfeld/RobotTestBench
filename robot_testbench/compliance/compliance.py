"""
Compliance testing module for simulating industry standard test procedures.
Implements environmental, endurance, and safety tests based on ISO, JESD, and UL standards.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

@dataclass
class EnvironmentalProfile:
    """Environmental test profile configuration."""
    name: str
    temperature_range: Tuple[float, float]  # (min, max) in °C
    humidity_range: Tuple[float, float]     # (min, max) in %
    vibration_profile: Dict[str, float]     # frequency (Hz) -> amplitude (g)
    duration: float                         # hours
    ramp_rate: float                        # °C/min

@dataclass
class DutyCycleProfile:
    """Duty cycle test profile configuration."""
    name: str
    load_profile: List[Tuple[float, float]]  # [(torque, duration), ...] in (N⋅m, s)
    cycle_count: int
    rest_period: float                       # seconds between cycles
    max_temperature: float                   # °C

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    start_time: datetime
    end_time: datetime
    passed: bool
    measurements: Dict[str, List[float]]
    failure_point: Optional[Tuple[str, float]] = None

class ComplianceTester:
    """Compliance testing simulator for robotic systems."""
    
    def __init__(self):
        # Standard test profiles
        self.environmental_profiles = {
            "ISO_16750_4": EnvironmentalProfile(
                name="ISO 16750-4 Temperature Cycling",
                temperature_range=(-40, 85),
                humidity_range=(10, 95),
                vibration_profile={},
                duration=1000,
                ramp_rate=1.0
            ),
            "JESD22_A104": EnvironmentalProfile(
                name="JESD22-A104 Temperature Cycling",
                temperature_range=(-55, 125),
                humidity_range=(0, 0),
                vibration_profile={},
                duration=500,
                ramp_rate=2.0
            ),
            "GMW3172": EnvironmentalProfile(
                name="GMW3172 Vibration",
                temperature_range=(23, 23),
                humidity_range=(50, 50),
                vibration_profile={
                    "10": 0.5,    # 10 Hz -> 0.5g
                    "20": 1.0,    # 20 Hz -> 1.0g
                    "50": 2.0,    # 50 Hz -> 2.0g
                    "100": 1.0,   # 100 Hz -> 1.0g
                },
                duration=24,
                ramp_rate=0.0
            )
        }
        
        self.duty_cycle_profiles = {
            "ISO_16750_2": DutyCycleProfile(
                name="ISO 16750-2 Endurance",
                load_profile=[
                    (10.0, 60),   # 10 N⋅m for 60s
                    (5.0, 30),    # 5 N⋅m for 30s
                    (0.0, 10)     # Rest for 10s
                ],
                cycle_count=10000,
                rest_period=10.0,
                max_temperature=85.0
            ),
            "UL_3300": DutyCycleProfile(
                name="UL 3300 Service Robot",
                load_profile=[
                    (8.0, 300),   # 8 N⋅m for 5min
                    (4.0, 60),    # 4 N⋅m for 1min
                    (0.0, 60)     # Rest for 1min
                ],
                cycle_count=5000,
                rest_period=60.0,
                max_temperature=75.0
            )
        }
    
    def run_environmental_test(self, 
                             profile: EnvironmentalProfile,
                             motor,
                             logger) -> TestResult:
        """
        Run environmental test simulation.
        
        Args:
            profile: Environmental test profile
            motor: Motor simulator instance
            logger: Data logger instance
            
        Returns:
            TestResult with test data and pass/fail status
        """
        start_time = datetime.now()
        measurements = {
            "temperature": [],
            "humidity": [],
            "vibration": [],
            "position": [],
            "velocity": [],
            "torque": [],
            "current": [],
            "power": []
        }
        
        # Simulate temperature cycling
        temp = profile.temperature_range[0]
        time = 0.0
        dt = 0.1  # 100ms time step
        
        while time < profile.duration * 3600:  # Convert hours to seconds
            # Update temperature
            if temp < profile.temperature_range[1]:
                temp += profile.ramp_rate * dt / 60  # Convert min to s
            else:
                temp = profile.temperature_range[0]
            
            # Simulate motor response to temperature
            motor.update_temperature(temp)
            
            # Record measurements
            measurements["temperature"].append(temp)
            measurements["humidity"].append(
                np.random.uniform(*profile.humidity_range)
            )
            measurements["vibration"].append(
                self._calculate_vibration(profile.vibration_profile, time)
            )
            measurements["position"].append(motor.get_position())
            measurements["velocity"].append(motor.get_velocity())
            measurements["torque"].append(motor.get_torque())
            measurements["current"].append(motor.get_current())
            measurements["power"].append(motor.get_power())
            
            # Log data
            logger.log({
                "time": time,
                "temperature": temp,
                "position": motor.get_position(),
                "velocity": motor.get_velocity(),
                "torque": motor.get_torque(),
                "current": motor.get_current(),
                "power": motor.get_power()
            })
            
            time += dt
            
            # Check for failures
            if self._check_failure(motor, temp):
                return TestResult(
                    test_name=profile.name,
                    start_time=start_time,
                    end_time=datetime.now(),
                    passed=False,
                    measurements=measurements,
                    failure_point=("temperature", temp)
                )
        
        return TestResult(
            test_name=profile.name,
            start_time=start_time,
            end_time=datetime.now(),
            passed=True,
            measurements=measurements
        )
    
    def run_duty_cycle_test(self,
                           profile: DutyCycleProfile,
                           motor,
                           logger) -> TestResult:
        """
        Run duty cycle endurance test simulation.
        
        Args:
            profile: Duty cycle test profile
            motor: Motor simulator instance
            logger: Data logger instance
            
        Returns:
            TestResult with test data and pass/fail status
        """
        start_time = datetime.now()
        measurements = {
            "cycle": [],
            "torque": [],
            "temperature": [],
            "position": [],
            "velocity": [],
            "current": [],
            "power": []
        }
        
        cycle = 0
        time = 0.0
        dt = 0.1  # 100ms time step
        
        while cycle < profile.cycle_count:
            # Run through load profile
            for torque, duration in profile.load_profile:
                end_time = time + duration
                
                while time < end_time:
                    # Apply load
                    motor.apply_torque(torque)
                    
                    # Record measurements
                    measurements["cycle"].append(cycle)
                    measurements["torque"].append(torque)
                    measurements["temperature"].append(motor.get_temperature())
                    measurements["position"].append(motor.get_position())
                    measurements["velocity"].append(motor.get_velocity())
                    measurements["current"].append(motor.get_current())
                    measurements["power"].append(motor.get_power())
                    
                    # Log data
                    logger.log({
                        "time": time,
                        "cycle": cycle,
                        "torque": torque,
                        "temperature": motor.get_temperature(),
                        "position": motor.get_position(),
                        "velocity": motor.get_velocity(),
                        "current": motor.get_current(),
                        "power": motor.get_power()
                    })
                    
                    time += dt
                    
                    # Check for failures
                    if self._check_failure(motor, motor.get_temperature()):
                        return TestResult(
                            test_name=profile.name,
                            start_time=start_time,
                            end_time=datetime.now(),
                            passed=False,
                            measurements=measurements,
                            failure_point=("cycle", cycle)
                        )
            
            # Rest period
            time += profile.rest_period
            cycle += 1
        
        return TestResult(
            test_name=profile.name,
            start_time=start_time,
            end_time=datetime.now(),
            passed=True,
            measurements=measurements
        )
    
    def _calculate_vibration(self,
                           profile: Dict[str, float],
                           time: float) -> float:
        """Calculate vibration amplitude at given time."""
        total = 0.0
        for freq, amp in profile.items():
            total += amp * np.sin(2 * np.pi * float(freq) * time)
        return total
    
    def _check_failure(self, motor, temperature: float) -> bool:
        """
        Check for test failure conditions.
        
        Failure conditions:
        - Temperature exceeds maximum
        - Current exceeds maximum
        - Position error exceeds threshold
        - Velocity error exceeds threshold
        """
        # Temperature check
        if temperature > 100.0:  # 100°C max
            return True
            
        # Current check
        if abs(motor.get_current()) > 10.0:  # 10A max
            return True
            
        # Position error check
        if abs(motor.get_position_error()) > 0.1:  # 0.1 rad max
            return True
            
        # Velocity error check
        if abs(motor.get_velocity_error()) > 1.0:  # 1 rad/s max
            return True
            
        return False 