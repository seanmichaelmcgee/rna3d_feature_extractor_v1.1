# RNA 3D Feature Extractor

A streamlined toolkit for extracting features from RNA molecules for machine learning applications.

## Features

- **Thermodynamic Analysis**: Extract energy landscapes, base-pairing probabilities, and structural features
- **Dihedral Analysis**: Calculate pseudo-dihedral angles from 3D structural data
- **Mutual Information Analysis**: Analyze evolutionary coupling signals from sequence alignments

## Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rna3d_feature_extractor.git
cd rna3d_feature_extractor

# Install dependencies
pip install -e .
```

### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rna3d_feature_extractor.git
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

- Python 3.8+
- ViennaRNA
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook
- BioPython
- forgi

## Usage

This package provides Jupyter notebooks for three different types of data:

1. `notebooks/train_features_extraction.ipynb`: For training data with 3D structures
2. `notebooks/validation_features_extraction.ipynb`: For validation data with 3D structures
3. `notebooks/test_features_extraction.ipynb`: For test data without 3D structures

## Directory Structure

```
rna3d_feature_extractor/
├── data/               # Data storage
│   ├── input/          # Input RNA files
│   ├── output/         # Extracted features
│   └── visualizations/ # Generated visualizations
├── notebooks/          # Jupyter notebooks for feature extraction
├── scripts/            # Utility scripts
└── src/                # Source code
    └── analysis/       # Analysis modules
        ├── thermodynamic_analysis.py
        ├── dihedral_analysis.py
        └── mutual_information.py
```

## Docker Usage

The included Dockerfile provides a containerized environment with all dependencies pre-installed. This ensures consistent behavior across different systems.

### Building the Docker Image

```bash
docker build -t rna3d-extractor .
```

### Running the Container

Basic usage with default parameters:

```bash
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  rna3d-extractor
```

Advanced usage with custom parameters:

```bash
docker run --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  -e JOBS=4 -e PF_SCALE=2.0 \
  rna3d-extractor
```

For interactive exploration:

```bash
docker run --rm -it \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/processed:/app/data/processed \
  --entrypoint /bin/bash \
  rna3d-extractor
```

For more information on Docker integration and testing strategies, see [docker-testing-strategy.md](docs/docker-testing-strategy.md).

## License

MIT