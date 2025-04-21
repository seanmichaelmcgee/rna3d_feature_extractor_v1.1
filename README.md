# RNA 3D Feature Extractor

A comprehensive toolkit for extracting structural, thermodynamic, and evolutionary features from RNA sequences.

## Overview

This toolkit extracts a variety of features from RNA sequences, including:

- Thermodynamic features (minimum free energy, ensemble energy, pairing probabilities)
- Mutual information features from sequence alignments
- Structural features for RNA 3D modeling
- Dihedral angle features

It provides a modular, memory-efficient architecture designed to handle large RNA sequences.

## Features

### Thermodynamic Analysis
- Extract energy landscapes and base-pairing probabilities
- Calculate positional entropy and structural features
- Uses ViennaRNA for accurate RNA secondary structure prediction

### Dihedral Analysis
- Calculate pseudo-dihedral angles from 3D structural data
- Process coordinate data from PDB or CSV sources
- Support for RNA residue-specific geometry analysis

### Mutual Information Analysis
- Analyze evolutionary coupling signals from Multiple Sequence Alignments (MSAs)
- **Pseudocount Correction**:
  - Adaptive pseudocount selection based on MSA size
  - Improves statistical robustness for sparse MSAs
  - Integrates with sequence weighting and APC correction
- RNA-specific Average Product Correction (APC)
- Chunking support for long sequences
- **Single-sequence MSA optimization**: Automatically skips MI calculation for single-sequence MSAs

## Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone <repository_url>
cd rna3d_feature_extractor

# Run the setup script (recommended)
./setup.sh

# Or manually create and activate the environment
mamba env create -f environment.yml
mamba activate rna3d-core

# Install in development mode
pip install -e .
```

For detailed instructions on environment setup, including troubleshooting common issues, see our [Environment Setup Guide](docs/environment-setup.md).

### Option 2: Docker Installation

```bash
# Clone the repository
git clone <repository_url>
cd rna3d_feature_extractor

# Build the Docker image
docker build -t rna3d-extractor .

# Run the container
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  rna3d-extractor
```

## Requirements

- Python 3.9+
- ViennaRNA 2.6.4+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook
- BioPython

## Usage

### Command-line Interface

Extract features for a single RNA target:

```bash
python scripts/run_feature_extraction_single.sh <target_id>
```

Process multiple targets in batch mode:

```bash
python -m src.workflow --targets-file data/targets.txt --extract-thermo --extract-mi
```

### Python API

```python
from src.workflow import RNAFeatureExtractionWorkflow

# Create workflow instance
workflow = RNAFeatureExtractionWorkflow()

# Extract features for a single target
results = workflow.extract_single_target(
    target_id="R1107",
    extract_thermo=True,
    extract_mi=True
)

# Process multiple targets
results = workflow.run_extraction(
    targets_file="data/targets.txt",
    extract_thermo=True,
    extract_mi=True
)
```

### Jupyter Notebooks

For interactive use, explore:

- `notebooks/refactored_workflow_demo.ipynb` - Demonstrates the refactored architecture
- `notebooks/test_features_extraction.ipynb` - For test data feature extraction
- `notebooks/train_features_extraction.ipynb` - For training data feature extraction
- `notebooks/validation_features_extraction.ipynb` - For validation data feature extraction

## Architecture

The toolkit follows a modular architecture:

- **DataManager**: Handles data loading, saving, and I/O operations
- **FeatureExtractor**: Manages feature extraction algorithms
- **BatchProcessor**: Coordinates processing multiple RNA targets
- **MemoryMonitor**: Tracks memory usage during processing
- **ResultValidator**: Ensures feature quality and compatibility

## Documentation

- [Feature Summary](FEATURE_SUMMARY.md) - Overview of extracted features
- [Shell Feature Extraction](SHELL_FEATURE_EXTRACTION.md) - Legacy shell script usage
- [Repository Transfer Guide](docs/REPOSITORY_TRANSFER.md) - Guide for transferring to a new repository

## Development Information

This project is currently under active refactoring. We're using multiple Git remotes to manage this process:

1. **Original Remote**: Contains the pre-refactored code
   - Reference name: `origin`

2. **Refactored Remote**: For testing the refactored code
   - Reference name: `refactor`
   - URL: `git@github.com:seanmichaelmcgee/rna3d_feature_extractor_v2.1.0_test.git`

For details on our remote repository strategy, see [REFACTOR_REMOTE_STRATEGY.md](docs/REFACTOR_REMOTE_STRATEGY.md).

For current refactoring status, see [REFACTORING_STATUS.md](REFACTORING_STATUS.md).

## Testing

```bash
# Activate the environment
mamba activate rna3d-core

# Run all tests
python -m unittest discover

# Run specific tests
python -m unittest tests.data.test_data_manager
```

For more information on testing, see [tests/README.md](tests/README.md).

## License

MIT