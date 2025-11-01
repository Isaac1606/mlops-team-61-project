"""
Setup script for Bike Sharing Demand Prediction Project.

This package implements an end-to-end MLOps pipeline for predicting
hourly bike rental demand using best practices including:
- Cookiecutter project structure
- Object-oriented design
- Scikit-Learn pipelines
- MLflow experiment tracking
- DVC data versioning
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Load dependencies from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
else:
    requirements = []

setup(
    name="bike-sharing-demand",
    version="0.1.0",
    author="Team 61",
    author_email="mlops-team-61@example.com",
    description="MLOps project for predicting hourly bike rental demand",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Isaac1606/mlops-team-61-project",
    packages=find_packages(exclude=["tests", "tests.*", "*.tests", "*.tests.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "bike-data=src.data.make_dataset:main",
            "bike-train=src.models.train_model:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
