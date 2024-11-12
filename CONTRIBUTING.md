# Contributing to WMDP Benchmark

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Create a new conda environment:
```
conda create -n wmdp python=3.12
conda activate wmdp
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Add docstrings for all functions and classes
- Keep line length to 80 characters
- Use black for code formatting

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run existing tests
6. Submit PR with description of changes

## Reporting Issues

When reporting issues, please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Python version and environment details