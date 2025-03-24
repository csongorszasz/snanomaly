# snanomaly

# Inspiration
This project is mainly inspired by [this](http://arxiv.org/abs/1905.11516) paper.

# Objective
Develop a pipeline that can most reliably detect anomalies in datasets of supernovae. The main focus is on the analysis of light curves, which are time series data representing the brightness of supernovae over time.

# Project Structure Description

## Setup
- `pyproject.toml`: Python project configuration
  - run `pip install -e ".[dev]"` to install the package and its dependencies

## Main Package Directory (`src/snanomaly/`)
- `dataset`: Data loading
- `models/`: Model classes
- `preprocessing/`: Data preprocessing and feature extraction
- `utils/`: Utility functions
- `visualization/`: Data visualization tools
- `scripts/`: Standalone scripts for development

## Supporting Directories
- `datasets/`: Supernova datasets
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training
- `tests/`: Test suite
