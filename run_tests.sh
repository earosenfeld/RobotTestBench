#!/bin/bash
# Test runner script for Robot Test Bench
# This ensures pytest runs with the correct Python path

echo "Running Robot Test Bench tests..."
python -m pytest "$@" 