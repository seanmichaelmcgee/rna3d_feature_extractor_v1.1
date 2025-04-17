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

#### Using the Notebook (Recommended)
```bash
jupyter notebook notebooks/train_features_extraction.ipynb
```
Set `LIMIT = None` to process all training data.

#### Using the Enhanced Shell Scripts (New)

We've implemented robust shell scripts for each data type that provide the same functionality as the notebooks with added features for production use:

```bash
# Process training data with all features
./scripts/feature_extraction/extract_train_features.sh

# With additional options
./scripts/feature_extraction/extract_train_features.sh \
  --csv data/raw/train_sequences.csv \
  --output-dir data/processed \
  --limit 5 \
  --cores 4 \
  --pf-scale 1.5 \
  --target 1SCL_A \
  --batch-size 5 \
  --memory-limit 16 \
  --verbose \
  --report
```

These scripts include:
- Memory monitoring and management
- Progress tracking and resumption capabilities
- Detailed error handling and logging
- Comprehensive HTML reporting
- Skip-existing functionality
- All feature types in a single workflow

For full documentation, see [scripts/feature_extraction/README.md](scripts/feature_extraction/README.md).

#### Using Legacy Command Line (Not Recommended)

The older command line approach has serious limitations compared to the notebooks and the new shell scripts:

```bash
# Thermodynamic features only (missing proper ViennaRNA integration)
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

### Processing Validation Data

For validation data with multiple coordinate structures:

#### Using the Notebook
```bash
jupyter notebook notebooks/validation_features_extraction.ipynb
```

#### Using the Enhanced Shell Script
```bash
./scripts/feature_extraction/extract_validation_features.sh

# With options for handling multiple coordinate sets
./scripts/feature_extraction/extract_validation_features.sh \
  --target R1107 \
  --verbose \
  --report
```

The validation script includes specialized handling for multiple coordinate sets:
- Proper filtering of empty coordinate sets (`-1e+18` placeholder values)
- Multi-structure output format that preserves all alternative structural coordinates
- Specialized ID matching for validation features

### Processing Test Data

Test data typically has no 3D structures, so only thermodynamic and MI features are extracted (no dihedral angles). 

#### Using the Notebook
```bash
jupyter notebook notebooks/test_features_extraction.ipynb
```

#### Using the Enhanced Shell Script
```bash
./scripts/feature_extraction/extract_test_features.sh

# With options
./scripts/feature_extraction/extract_test_features.sh \
  --target R1107 \
  --verbose \
  --report
```

### Legacy Command Line Scripts

The following legacy scripts are still available but not recommended:

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
├── data/                          # Data storage
│   ├── raw/                       # Input RNA files
│   └── processed/                 # Extracted features
│       ├── dihedral_features/     # Dihedral angle features
│       ├── mi_features/           # Mutual information features
│       └── thermo_features/       # Thermodynamic features
├── docs/                          # Documentation
│   ├── docker-testing-strategy.md # Docker testing documentation
│   ├── environment-setup.md       # Detailed environment setup guide
│   └── feature-renaming.md        # Feature naming guidelines
├── notebooks/                     # Jupyter notebooks
│   ├── train_features_extraction.ipynb    # Training data extraction
│   ├── validation_features_extraction.ipynb # Validation data extraction
│   ├── test_features_extraction.ipynb     # Test data extraction
│   └── mi_pseudocount_demo.ipynb          # Demonstrates pseudocounts
├── scripts/                       # Utility scripts
│   ├── feature_extraction/        # Enhanced shell scripts
│   │   ├── extract_train_features.sh    # Training data extraction
│   │   ├── extract_validation_features.sh # Validation data extraction
│   │   ├── extract_test_features.sh     # Test data extraction
│   │   ├── verify_features.py           # Feature verification
│   │   ├── test_extraction_scripts.sh   # Simple test script
│   │   └── README.md                    # Script documentation
│   ├── compare_docker_outputs.py        # Docker output comparison
│   ├── extract_pseudodihedral_features.py # Dihedral extraction
│   ├── run_dihedral_extraction.sh       # Run dihedral extraction
│   ├── run_feature_extraction_single.sh # Single target extraction
│   ├── run_mi_batch.sh                  # MI batch processing
│   ├── single_target_test.py            # Single target test
│   ├── test_load_structure.py           # Structure loading test
│   └── verify_feature_compatibility.py  # Feature compatibility check
├── src/                           # Source code
│   ├── analysis/                  # Analysis modules
│   │   ├── dihedral_analysis.py
│   │   ├── memory_monitor.py
│   │   ├── mutual_information.py
│   │   ├── rna_mi_pipeline/       # Enhanced MI implementation
│   │   │   ├── enhanced_mi.py
│   │   │   ├── mi_config.py
│   │   │   ├── rna_mi_pipeline.py
│   │   │   └── tech_guide.md
│   │   └── thermodynamic_analysis.py
│   └── data/                      # Data processing utilities
│       ├── batch_feature_runner.py
│       ├── extract_features_simple.py
│       ├── npz_to_csv.py
│       └── visualize_features.py
└── tests/                         # Unit tests
    ├── analysis/                  # Analysis module tests
    │   ├── test_feature_names.py
    │   ├── test_mi_pseudocounts.py
    │   └── test_thermodynamic_analysis.py
    ├── test_feature_extraction_scripts.sh # Comprehensive test runner
    └── README.md                  # Test documentation
```

## Memory Optimization

This toolkit includes memory monitoring and optimization tools for running in constrained environments:

- Chunking of large RNA sequences
- Adaptive batch processing for MSAs
- Memory-aware parameter selection
- Resource usage tracking
- Dynamic batch size adjustment in shell scripts

## Testing

This toolkit includes comprehensive testing infrastructure:

### Unit Tests

```bash
# Run all unit tests
python -m unittest discover tests

# Run specific test module
python -m unittest tests.analysis.test_mi_pseudocounts
```

### Feature Extraction Tests

The enhanced shell scripts include their own testing infrastructure:

```bash
# Run comprehensive testing of all shell scripts
./tests/test_feature_extraction_scripts.sh

# This will run various tests and generate a detailed report:
# - Basic functionality tests for all data types
# - Resume capability tests
# - Skip-existing functionality tests
# - Batch processing tests
# - Feature verification tests
```

The test output includes:
- Detailed logs in `tests/feature_extraction_test.log`
- Markdown report in `tests/feature_extraction_test_report.md`
- Test output files in `data/test_output`

### Feature Verification

You can verify the format of extracted feature files:

```bash
# Verify features in a directory
./scripts/feature_extraction/verify_features.py data/processed

# Verify specific feature files
./scripts/feature_extraction/verify_features.py . \
  --thermo-file data/processed/thermo_features/1SCL_A_thermo_features.npz
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