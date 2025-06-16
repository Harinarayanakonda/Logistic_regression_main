#!/bin/bash

# Test runner script for the data preprocessing pipeline

# Exit on error
set -e

echo "Running tests with pytest..."
python -m pytest tests/ -v --cov=./ --cov-report=xml

echo "Running mypy for type checking..."
mypy .

echo "Running flake8 for linting..."
flake8 .

echo "All checks passed!"