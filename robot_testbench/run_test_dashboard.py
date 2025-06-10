#!/usr/bin/env python3
"""
Run the RobotTestBench test results dashboard.
"""

import argparse
from robot_testbench.visualization.test_results import TestResultsDashboard

def main():
    parser = argparse.ArgumentParser(description="Run RobotTestBench test results dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--data-dir", type=str, default="data/logs", help="Directory containing test data")
    
    args = parser.parse_args()
    
    dashboard = TestResultsDashboard(data_dir=args.data_dir)
    print(f"Starting dashboard on http://localhost:{args.port}")
    dashboard.run_server(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main() 