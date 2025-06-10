"""
Test runner script for executing different types of tests.
"""

import argparse
import sys
from pathlib import Path
import pytest

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Robot Test Bench tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        help="Specific test protocol to run"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )
    return parser.parse_args()

def run_tests(args):
    """Run the specified tests."""
    # Base pytest arguments
    pytest_args = ["-v" if args.verbose else ""]
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(["--cov=robot_testbench", "--cov-report=term-missing"])
        if args.html:
            pytest_args.extend(["--cov-report=html:reports/coverage"])
    
    # Add benchmark if requested
    if args.benchmark:
        pytest_args.extend(["--benchmark-only"])
    
    # Select test type
    if args.type == "unit":
        pytest_args.append("tests/unit")
    elif args.type == "integration":
        pytest_args.append("tests/integration")
    elif args.type == "performance":
        pytest_args.append("tests/performance")
    else:  # all
        pytest_args.append("tests")
    
    # Add specific protocol if specified
    if args.protocol:
        protocol_path = Path("tests/protocols") / args.protocol
        if not protocol_path.exists():
            print(f"Error: Test protocol {args.protocol} not found")
            sys.exit(1)
        pytest_args.append(str(protocol_path))
    
    # Run tests
    return pytest.main(pytest_args)

def main():
    """Main entry point."""
    args = parse_args()
    return run_tests(args)

if __name__ == "__main__":
    sys.exit(main()) 