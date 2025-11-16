# Makefile for Bike Sharing Demand Prediction Project
# Provides commands for common tasks with full reproducibility

.PHONY: help install install-dev clean data train evaluate all test lint format mlflow-ui dvc-pull dvc-push

# Default target
help:
	@echo "Bike Sharing Demand Prediction - MLOps Project"
	@echo "================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make clean           - Clean generated files"
	@echo "  make data            - Run data processing pipeline"
	@echo "  make train           - Train models"
	@echo "  make evaluate        - Evaluate trained models"
	@echo "  make all             - Run full pipeline (data + train)"
	@echo "  make mlflow-ui       - Start MLflow UI"
	@echo "  make dvc-pull        - Pull data from DVC remote"
	@echo "  make dvc-push        - Push data to DVC remote"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linter"
	@echo "  make format          - Format code"

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install -e .
	pip install black flake8 pytest pytest-cov

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pkl" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Clean complete"

# Data processing
data:
	@echo "Running data processing pipeline..."
	python src/data/make_dataset.py

# Model training
train:
	@echo "Training models..."
	python src/models/train_model.py

# Evaluation (placeholder - can be extended)
evaluate:
	@echo "Evaluating models..."
	@echo "Run 'make mlflow-ui' to view detailed evaluation metrics"

# Full pipeline
all: clean data train
	@echo "Full pipeline complete!"

# MLflow UI
mlflow-ui:
	@echo "Starting MLflow UI..."
	mlflow ui --backend-store-uri file://$(PWD)/mlruns

# DVC commands
dvc-pull:
	@echo "Pulling data from DVC remote..."
	dvc pull -r raw --force

dvc-push:
	@echo "Pushing data to DVC remote..."
	dvc push -r raw

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-drift:
	@echo "Running data drift tests..."
	pytest tests/data_drift/ -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	@echo "Running fast tests (unit only)..."
	pytest tests/unit/ -v -x

# Linting
lint:
	@echo "Running linter..."
	flake8 src/ --max-line-length=100 --exclude=__pycache__

# Code formatting
format:
	@echo "Formatting code..."
	black src/ --line-length=100

