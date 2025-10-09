from setuptools import setup, find_packages
# Load dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
# Define package metadata and configuration
setup(
    name="mlops_project",          # Package name
    version="0.1",             
    author="Team61",
    packages=find_packages(),      # Automatically discover all sub-packages
    install_requires=requirements, # External dependencies listed 
)