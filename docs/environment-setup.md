# Environment Setup Guide

This document provides detailed instructions for setting up the computational environment required for running the RNA 3D Feature Extractor.

## Prerequisites

- **Operating System**: Linux or macOS (Windows users should use WSL2)
- **Mamba or Conda**: We strongly recommend using [Mamba](https://github.com/conda-forge/miniforge#mambaforge) for faster dependency resolution

## Quick Setup

The simplest way to set up the environment is to run the setup script from the project root:

```bash
./setup.sh
```

This script will:
1. Check for Mamba/Conda installation
2. Create or update the `rna3d-core` environment
3. Verify critical dependencies (ViennaRNA, PyTorch, etc.)
4. Make all necessary scripts executable

## Manual Setup

If you prefer to set up the environment manually, follow these steps:

1. **Install Mamba** (recommended) or Conda if you haven't already:
   ```bash
   # Install Mamba (if you have Conda already)
   conda install mamba -n base -c conda-forge
   
   # Or download and install Mambaforge from:
   # https://github.com/conda-forge/miniforge#mambaforge
   ```

2. **Create the environment** from the provided configuration:
   ```bash
   # Using Mamba (recommended)
   mamba env create -f environment.yml
   
   # Or using Conda
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   mamba activate rna3d-core
   # Or
   conda activate rna3d-core
   ```

4. **Verify the installation**:
   ```bash
   # Test ViennaRNA
   python -c "import RNA; print(f'ViennaRNA version: {RNA.__version__}')"
   
   # Test other core dependencies
   python -c "import numpy, pandas, matplotlib, torch; print('Core dependencies available')"
   ```

## Critical Dependencies

The following packages are essential for the RNA 3D Feature Extractor:

- **ViennaRNA**: Required for RNA secondary structure prediction
- **PyTorch**: Used for neural network models
- **BioPython**: Handles biological sequences and structures
- **NumPy/Pandas**: Core data handling

## Troubleshooting

If you encounter problems during environment setup:

1. **ViennaRNA Installation Issues**:
   - On Ubuntu/Debian: `sudo apt-get install libviennarna-dev`
   - On macOS: `brew install viennarna`
   - Then try: `pip install forgi`

2. **Environment Creation Fails**:
   ```bash
   # Try cleaning your conda cache first
   mamba clean --all
   # Then recreate the environment
   mamba env create -f environment.yml
   ```

3. **Package Conflicts**:
   ```bash
   # Remove the problematic environment
   mamba env remove -n rna3d-core
   # Recreate with more flexible version constraints
   # (edit environment.yml to remove specific version numbers)
   mamba env create -f environment.yml
   ```

4. **General Issues**:
   - Make sure you're running commands from the project root directory
   - Check you have sufficient disk space and permissions
   - Consider updating Mamba/Conda: `conda update -n base conda`

## Running with Docker

For reproducible environments across different systems, you can use the provided Docker configuration:

```bash
# Build the Docker image
docker build -t rna3d-feature-extractor .

# Run a container with the current directory mounted
docker run -it --rm -v $(pwd):/app rna3d-feature-extractor
```

See the [Docker testing strategy](docker-testing-strategy.md) document for more details on Docker usage.