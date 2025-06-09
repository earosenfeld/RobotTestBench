#!/usr/bin/env python3
"""
RobotTestBench - Main Application Entry Point
"""

import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure all required directories exist."""
    directories = ['data', 'data/logs', 'data/tests']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RobotTestBench - Actuator and Sensor Testing Platform')
    parser.add_argument('--mode', choices=['cli', 'gui'], default='gui',
                      help='Application mode: cli for command line, gui for graphical interface')
    parser.add_argument('--config', type=str, default='config/default.json',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main application entry point."""
    args = parse_args()
    setup_directories()
    
    logger.info("Starting RobotTestBench...")
    
    if args.mode == 'gui':
        try:
            from visualization.dashboard import launch_dashboard
            launch_dashboard()
        except ImportError:
            logger.error("Failed to import dashboard module. Make sure all dependencies are installed.")
            return 1
    else:
        # CLI mode implementation will go here
        logger.info("CLI mode not yet implemented")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 