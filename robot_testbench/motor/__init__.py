"""
Motor simulation and control module.
"""

from .motor_simulator import MotorSimulator, MotorParameters
from .dyno import DynoSimulator, DynoParameters
from .load_motor import LoadMotor, LoadMotorParameters, Load
from .electrical import (
    PowerSupply,
    PowerSupplyConfig,
    I2CInterface,
    I2CConfig,
    FaultInjector,
    FaultConfig
)

__all__ = [
    'MotorSimulator',
    'MotorParameters',
    'DynoSimulator',
    'DynoParameters',
    'LoadMotor',
    'LoadMotorParameters',
    'Load',
    'PowerSupply',
    'PowerSupplyConfig',
    'I2CInterface',
    'I2CConfig',
    'FaultInjector',
    'FaultConfig'
] 