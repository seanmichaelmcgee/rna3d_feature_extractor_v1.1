# RNA 3D Feature Extractor

A streamlined toolkit for extracting features from RNA molecules for machine learning applications.

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
  - Adaptive pseudocount selection based on MSA size:
    - Small MSAs (≤25 sequences): 0.5
    - Medium MSAs (26-100 sequences): 0.2
    - Large MSAs (>100 sequences): 0.0
  - Improves statistical robustness for sparse MSAs (~10% of dataset)
  - Integrates with sequence weighting and APC correction
  - Can be manually specified or automatically selected
- Sequence weighting to reduce redundancy bias
- RNA-specific Average Product Correction (APC)
- Chunking support for long sequences (>750 nucleotides)

## Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rna3d_feature_extractor.git
cd rna3d_feature_extractor

# Create and activate conda environment
mamba env create -f environment.yml
mamba activate rna3d-core

# Install in development mode
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
- ViennaRNA 2.6.4+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook
- BioPython
- forgi

## Usage

### Feature Extraction Notebooks

This package provides Jupyter notebooks for three different types of data:

1. `notebooks/train_features_extraction.ipynb`: For training data with 3D structures
2. `notebooks/validation_features_extraction.ipynb`: For validation data with 3D structures
3. `notebooks/test_features_extraction.ipynb`: For test data without 3D structures

### Demonstration Notebooks

- `notebooks/mi_pseudocount_demo.ipynb`: Demonstrates pseudocount improvement on MI calculations

### Processing Training Data

Training data with 3D structures can be processed using either the notebook or command line:

#### Using the Notebook
```bash
jupyter notebook notebooks/train_features_extraction.ipynb
```
Set `LIMIT = None` to process all training data.

#### Using Command Line (Recommended for Large Datasets)
```bash
python src/data/batch_feature_runner.py \
  --csv data/raw/train_sequences.csv \
  --output-dir data/processed/features \
  --id-col target_id \
  --seq-col sequence \
  --length-min 1 \
  --length-max 10000 \
  --batch-size 20 \
  --dynamic-batch \
  --retry 2 \
  --pf-scale 1.5 \
  --verbose \
  --checkpoint-interval 10 \
  --resume
```

This command provides robust error recovery and memory management for processing large datasets.

### Processing Validation Data

For validation data with multiple coordinate structures, use the validation notebook:

```bash
jupyter notebook notebooks/validation_features_extraction.ipynb
```

**Important Note**: The validation notebook contains specialized handling for multiple coordinate sets per structure that is not available in the command-line tools. For validation data specifically, we recommend using the notebook with `LIMIT = None` rather than the command-line tools.

Key validation data capabilities:
- Proper filtering of empty coordinate sets (`-1e+18` placeholder values)
- Multi-structure output format that preserves all alternative structural coordinates
- Specialized ID matching for validation features

### Processing Test Data

Test data typically has no 3D structures, so only thermodynamic features can be extracted (no dihedral angles). 

#### Using the Notebook
```bash
jupyter notebook notebooks/test_features_extraction.ipynb
```

#### Using Command Line
```bash
python src/data/batch_feature_runner.py \
  --csv data/raw/test_sequences.csv \
  --output-dir data/processed/test_features \
  --id-col target_id \
  --seq-col sequence \
  --length-min 1 \
  --length-max 10000 \
  --batch-size 20 \
  --dynamic-batch \
  --retry 2 \
  --pf-scale 1.5 \
  --verbose
```

Since test data does not include 3D structures, you'll only get thermodynamic features. If your test data does have coordinate information, you should use the training data extraction pipeline instead.

### Other Command Line Scripts

Extract pseudodihedral features:
```bash
python scripts/extract_pseudodihedral_features.py --target <target_id>
python scripts/extract_pseudodihedral_features.py --all  # Process all targets
```

Run MI calculation in batch mode:
```bash
python scripts/run_mi_batch.sh <target_list_file>
```

Process features for a single target:
```bash
python scripts/run_feature_extraction_single.sh <target_id>
```

## API Usage

### Mutual Information with Pseudocounts

```python
from src.analysis.mutual_information import calculate_mutual_information

# With adaptive pseudocount selection (recommended)
mi_result = calculate_mutual_information(msa_sequences, pseudocount=None)

# With explicit pseudocount value
mi_result = calculate_mutual_information(msa_sequences, pseudocount=0.5)

# Without pseudocount (original behavior)
mi_result = calculate_mutual_information(msa_sequences, pseudocount=0.0)
```

### Enhanced MI with Sequence Weighting and Pseudocounts

```python
from src.analysis.rna_mi_pipeline.enhanced_mi import calculate_mutual_information_enhanced

# With all enhancements
mi_result = calculate_mutual_information_enhanced(
    msa_sequences,
    pseudocount=None,  # Adaptive selection
    parallel=True
)
```

### Processing Complete MSA Files

```python
from src.analysis.rna_mi_pipeline.enhanced_mi import process_rna_msa_for_structure

# Process an MSA file with all optimizations
features = process_rna_msa_for_structure(
    msa_file="path/to/msa.fasta",
    output_features="path/to/output.npz",
    pseudocount=None,  # Adaptive selection
    parallel=True
)
```

## Configuration

The MI pipeline is configurable through `src/analysis/rna_mi_pipeline/mi_config.py`:

- MSA quality presets (high, medium, low) with appropriate pseudocount values
- Hardware-specific optimizations for different resource constraints
- RNA length-specific parameters for optimal chunking

## Directory Structure

```
rna3d_feature_extractor/
├── data/                         # Data storage
│   ├── raw/                      # Input RNA files
│   └── processed/                # Extracted features
│       ├── dihedral_features/    # Dihedral angle features
│       ├── mi_features/          # Mutual information features
│       └── thermo_features/      # Thermodynamic features
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Utility scripts
├── src/                          # Source code
│   ├── analysis/                 # Analysis modules
│   │   ├── dihedral_analysis.py
│   │   ├── memory_monitor.py
│   │   ├── mutual_information.py
│   │   ├── rna_mi_pipeline/      # Enhanced MI implementation
│   │   │   ├── enhanced_mi.py
│   │   │   ├── mi_config.py
│   │   │   └── rna_mi_pipeline.py
│   │   └── thermodynamic_analysis.py
│   └── data/                     # Data processing utilities
└── tests/                        # Unit tests
```

## Memory Optimization

This toolkit includes memory monitoring and optimization tools for running in constrained environments:

- Chunking of large RNA sequences
- Adaptive batch processing for MSAs
- Memory-aware parameter selection
- Resource usage tracking

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