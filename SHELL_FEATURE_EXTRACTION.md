# Shell-Based Feature Extraction

This document outlines a plan for implementing consistent shell-based feature extraction that mirrors the functionality of the Jupyter notebooks. The goal is to provide reliable command-line alternatives for all three types of feature extraction across all three datasets.

## Current Issues with Shell Scripts

1. **ViennaRNA Integration Issues**: The scripts fail to properly import ViennaRNA even when it's installed in the environment
2. **Feature Fragmentation**: Unlike notebooks which extract all features in one workflow:
   - `batch_feature_runner.py` only extracts thermodynamic features
   - MI features require separately running `run_mi_batch.sh`
   - Dihedral features require separately running `run_dihedral_extraction.sh`
3. **Missing Integrated Workflow**: No single script combines all feature types like the notebooks do
4. **Validation Data Complexity**: The validation notebook contains specialized handling for multiple coordinate sets that is missing from the shell scripts

## Implementation Plan

We will create three new shell scripts, one for each data type, that closely mirror the notebook functionality:

1. `extract_train_features.sh` - For training data with 3D structures
2. `extract_validation_features.sh` - For validation data with multiple coordinate sets
3. `extract_test_features.sh` - For test data without 3D structures

Each script will integrate all three types of feature extraction (thermodynamic, dihedral, and mutual information) as appropriate for the data type.

## Common Script Structure

Each script will follow this common structure:

```bash
#!/bin/bash
# Extract all features for [DATA_TYPE] data
# This script mirrors the functionality of the [NOTEBOOK_NAME] notebook
#
# Usage: ./extract_[DATA_TYPE]_features.sh [OPTIONS]
#
# Options:
#   --csv FILE          CSV file with sequences (default: data/raw/[DATA_TYPE]_sequences.csv)
#   --output-dir DIR    Output directory (default: data/processed)
#   --limit N           Limit processing to N targets (for testing)
#   --cores N           Number of cores to use (default: all - 1)
#   --pf-scale SCALE    Partition function scaling factor (default: 1.5)
#   --target ID         Process only the specified target
#   --verbose           Enable verbose output
#   --force             Overwrite existing output files
#   --help              Show this help message

# Set default parameters
CSV_FILE="data/raw/[DATA_TYPE]_sequences.csv"
OUTPUT_DIR="data/processed"
LIMIT=0  # 0 means no limit
CORES=$(( $(nproc) - 1 ))
PF_SCALE=1.5
TARGET=""
VERBOSE=false
FORCE=false

# Parse command-line arguments
# [Argument parsing code here]

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rna3d-core

echo "=== RNA [DATA_TYPE] Feature Extraction ==="
echo "Starting extraction at $(date)"
echo "  CSV file: $CSV_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Cores: $CORES"
echo "  PF scale: $PF_SCALE"
[Other parameter display]

# Create output directories
mkdir -p "$OUTPUT_DIR/thermo_features"
mkdir -p "$OUTPUT_DIR/dihedral_features"  # Only for train and validation
mkdir -p "$OUTPUT_DIR/mi_features"
mkdir -p "$OUTPUT_DIR/logs"

# Create data processing functions copied from the notebooks
# [Python functions copied from notebooks]

# Main processing code
# [Main code here]

echo "Feature extraction completed at $(date)"
```

## Detailed Implementation: Training Data Script

```bash
#!/bin/bash
# Extract all features for TRAINING data
# This script mirrors the functionality of the train_features_extraction.ipynb notebook
#
# Usage: ./extract_train_features.sh [OPTIONS]
#
# [Options as shown in common structure]

# [Common setup as shown above]

# Define Python script to run the feature extraction
cat > "$OUTPUT_DIR/logs/train_extraction_script.py" << 'PYTHONSCRIPT'
#!/usr/bin/env python3
# This script is auto-generated from the train_features_extraction.ipynb notebook

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import psutil

# Import feature extraction modules
from src.analysis.thermodynamic_analysis import extract_thermodynamic_features
from src.analysis.dihedral_analysis import extract_dihedral_features
from src.analysis.mutual_information import calculate_mutual_information, convert_mi_to_evolutionary_features
from src.data.extract_features_simple import save_features_npz
from src.analysis.memory_monitor import MemoryTracker, log_memory_usage

# Imported helper functions from notebook
def load_rna_data(csv_path):
    """Load RNA data from CSV file."""
    # [Function code copied from notebook]

def get_unique_target_ids(df, id_col="ID"):
    """Extract unique target IDs from dataframe."""
    # [Function code copied from notebook]

def load_structure_data(target_id, data_dir="data/raw"):
    """Load structure data for a given target from labels CSV."""
    # [Function code copied from notebook]

def load_msa_data(target_id, data_dir="data/raw"):
    """Load MSA data for a given target."""
    # [Function code copied from notebook]

def get_sequence_for_target(target_id, data_dir="data/raw"):
    """Get RNA sequence for a target ID from the sequence file."""
    # [Function code copied from notebook]

def extract_thermo_features_for_target(target_id, sequence=None):
    """Extract thermodynamic features for a given target."""
    # [Function code copied from notebook]

def extract_dihedral_features_for_target(target_id, structure_data=None):
    """Extract pseudodihedral angle features for a given target."""
    # [Function code copied from notebook]

def extract_mi_features_for_target(target_id, structure_data=None, msa_sequences=None):
    """Extract Mutual Information features for a given target."""
    # [Function code copied from notebook]

def process_target(target_id, extract_thermo=True, extract_dihedral=True, extract_mi=True):
    """Process a single target, extracting all requested feature types."""
    # [Function code copied from notebook]

def batch_process_targets(target_ids, extract_thermo=True, extract_dihedral=True, extract_mi=True):
    """Process multiple targets in batch mode."""
    # [Function code copied from notebook]

# Main code
import argparse
parser = argparse.ArgumentParser(description="Extract features for training data")
parser.add_argument("--csv", type=str, required=True, help="CSV file with sequences")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
parser.add_argument("--limit", type=int, default=0, help="Limit processing to N targets")
parser.add_argument("--target", type=str, default="", help="Process only the specified target")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

# Define data directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = Path(args.output_dir)

# Output directories for each feature type
THERMO_DIR = PROCESSED_DIR / "thermo_features"
DIHEDRAL_DIR = PROCESSED_DIR / "dihedral_features"
MI_DIR = PROCESSED_DIR / "mi_features"

# Make sure all directories exist
for directory in [RAW_DIR, PROCESSED_DIR, THERMO_DIR, DIHEDRAL_DIR, MI_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Set parameters
LIMIT = args.limit if args.limit > 0 else None  # None means no limit
VERBOSE = args.verbose

# Load data
train_labels_file = RAW_DIR / "train_labels.csv"
train_sequences_file = Path(args.csv)

train_labels = load_rna_data(train_labels_file)
train_sequences = load_rna_data(train_sequences_file)

if train_labels is None:
    print("Error loading training labels. Please make sure train_labels.csv exists.")
    sys.exit(1)
elif train_sequences is None:
    print(f"Error loading training sequences from {train_sequences_file}")
    sys.exit(1)

# Get unique target IDs
target_ids = get_unique_target_ids(train_labels)

# Verify sequences exist for target IDs
seq_id_col = next((col for col in ["target_id", "ID", "id"] if col in train_sequences.columns), None)
if seq_id_col:
    available_targets = set(train_sequences[seq_id_col])
    target_with_sequences = [tid for tid in target_ids if tid in available_targets]
    missing_sequences = len(target_ids) - len(target_with_sequences)
    
    if missing_sequences > 0:
        print(f"Warning: {missing_sequences} targets do not have sequences in {train_sequences_file}")
    
    print(f"Found sequences for {len(target_with_sequences)}/{len(target_ids)} targets")
    target_ids = target_with_sequences

# Process specific target if requested
if args.target:
    if args.target in target_ids:
        target_ids = [args.target]
        print(f"Processing only target: {args.target}")
    else:
        print(f"Target {args.target} not found in available targets")
        sys.exit(1)

# Apply limit if specified
if LIMIT is not None and LIMIT < len(target_ids):
    print(f"Limiting to first {LIMIT} targets for testing")
    target_ids = target_ids[:LIMIT]

# Process targets
results = batch_process_targets(
    target_ids,
    extract_thermo=True,
    extract_dihedral=True,
    extract_mi=True
)

# Verify features
print("\nVerifying processed features for compatibility...")
import subprocess
verification_script = Path("scripts/verify_feature_compatibility.py")
if verification_script.exists():
    try:
        cmd = [sys.executable, str(verification_script), str(PROCESSED_DIR), "--verbose"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Verification failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
    except Exception as e:
        print(f"Error running verification script: {e}")
else:
    print(f"Warning: Feature verification script not found at {verification_script}")

# Done
print("Feature extraction complete")
PYTHONSCRIPT

# Make the script executable
chmod +x "$OUTPUT_DIR/logs/train_extraction_script.py"

# Run the Python script
echo "Running training feature extraction..."
python "$OUTPUT_DIR/logs/train_extraction_script.py" \
  --csv "$CSV_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --verbose

# Print completion message
echo "===== Training Feature Extraction Complete ====="
echo "Output saved to $OUTPUT_DIR"
echo "Log saved to $OUTPUT_DIR/logs"
```

## Detailed Implementation: Validation Data Script

```bash
#!/bin/bash
# Extract all features for VALIDATION data
# This script mirrors the functionality of the validation_features_extraction.ipynb notebook
#
# Usage: ./extract_validation_features.sh [OPTIONS]
#
# [Options as shown in common structure]

# [Common setup as shown above]

# Define Python script to run the feature extraction
cat > "$OUTPUT_DIR/logs/validation_extraction_script.py" << 'PYTHONSCRIPT'
#!/usr/bin/env python3
# This script is auto-generated from the validation_features_extraction.ipynb notebook

# [Import statements as in the training script]

# Imported helper functions from the validation notebook
# IMPORTANT: These include the special validation-specific functions for handling multiple coordinate sets

def load_rna_data(csv_path):
    """Load RNA data from CSV file."""
    # [Function code copied from validation notebook]

def get_unique_target_ids(df, id_col="ID"):
    """Extract unique target IDs from dataframe."""
    # [Function code copied from validation notebook]

def load_structure_data(target_id, data_dir="data/raw"):
    """Load structure data for a given target from validation labels CSV."""
    # [Function code copied from validation notebook - includes validation-specific logic]

def load_msa_data(target_id, data_dir="data/raw"):
    """Load MSA data for a given target."""
    # [Function code copied from validation notebook]

def get_sequence_for_target(target_id, data_dir="data/raw"):
    """Get RNA sequence for a target ID from the sequence file."""
    # [Function code copied from validation notebook]

def extract_thermo_features_for_target(target_id, sequence=None):
    """Extract thermodynamic features for a given target."""
    # [Function code copied from validation notebook]

def extract_dihedral_features_for_target(target_id, structure_data=None):
    """Extract pseudodihedral angle features for a given target with multiple coordinate sets."""
    # [Function code copied from validation notebook - includes multi-coordinate handling]

def extract_mi_features_for_target(target_id, structure_data=None, msa_sequences=None):
    """Extract Mutual Information features for a given target."""
    # [Function code copied from validation notebook]

def process_target(target_id, extract_thermo=True, extract_dihedral=True, extract_mi=True):
    """Process a single target, extracting all requested feature types."""
    # [Function code copied from validation notebook]

def batch_process_targets(target_ids, extract_thermo=True, extract_dihedral=True, extract_mi=True):
    """Process multiple targets in batch mode."""
    # [Function code copied from validation notebook]

# Main code - similar to training script but using validation data paths
# [Validation-specific main code]
PYTHONSCRIPT

# Make the script executable
chmod +x "$OUTPUT_DIR/logs/validation_extraction_script.py"

# Run the Python script
echo "Running validation feature extraction..."
python "$OUTPUT_DIR/logs/validation_extraction_script.py" \
  --csv "$CSV_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --verbose

# Print completion message
echo "===== Validation Feature Extraction Complete ====="
echo "Output saved to $OUTPUT_DIR"
echo "Log saved to $OUTPUT_DIR/logs"
```

## Detailed Implementation: Test Data Script

```bash
#!/bin/bash
# Extract all features for TEST data
# This script mirrors the functionality of the test_features_extraction.ipynb notebook
#
# Usage: ./extract_test_features.sh [OPTIONS]
#
# [Options as shown in common structure]

# [Common setup as shown above]

# Define Python script to run the feature extraction
cat > "$OUTPUT_DIR/logs/test_extraction_script.py" << 'PYTHONSCRIPT'
#!/usr/bin/env python3
# This script is auto-generated from the test_features_extraction.ipynb notebook

# [Import statements as in the training script]

# Note: For test data, we only extract thermodynamic features
# as there are no 3D structures available

# Imported helper functions from the test notebook
def load_rna_data(csv_path):
    """Load RNA data from CSV file."""
    # [Function code copied from test notebook]

def get_unique_target_ids(df, id_col="ID"):
    """Extract unique target IDs from dataframe."""
    # [Function code copied from test notebook]

def get_sequence_for_target(target_id, data_dir="data/raw"):
    """Get RNA sequence for a target ID from the sequence file."""
    # [Function code copied from test notebook]

def extract_thermo_features_for_target(target_id, sequence=None):
    """Extract thermodynamic features for a given target."""
    # [Function code copied from test notebook]

def process_target(target_id, extract_thermo=True):
    """Process a single target, extracting thermodynamic features."""
    # [Function code copied from test notebook - simplified for thermo only]

def batch_process_targets(target_ids, extract_thermo=True):
    """Process multiple targets in batch mode."""
    # [Function code copied from test notebook - simplified for thermo only]

# Main code - simplified for test data (thermo features only)
# [Test-specific main code]
PYTHONSCRIPT

# Make the script executable
chmod +x "$OUTPUT_DIR/logs/test_extraction_script.py"

# Run the Python script
echo "Running test feature extraction..."
python "$OUTPUT_DIR/logs/test_extraction_script.py" \
  --csv "$CSV_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --verbose

# Print completion message
echo "===== Test Feature Extraction Complete ====="
echo "Output saved to $OUTPUT_DIR"
echo "Log saved to $OUTPUT_DIR/logs"
```

## Implementation Strategy

1. **Direct Code Translation**: The Python code in each script will be a direct copy of the corresponding notebook, ensuring identical functionality.

2. **ViennaRNA Integration Fix**: The scripts will properly import and use ViennaRNA by ensuring:
   - The right conda environment is activated before script execution
   - The import paths are correctly set
   - Any necessary environment variables are set

3. **Feature Integration**: Each script will handle all applicable feature types:
   - Training: thermodynamic, dihedral, and MI features
   - Validation: thermodynamic, dihedral (with multi-coordinate handling), and MI features
   - Test: thermodynamic features only

4. **Error Handling and Logging**: Each script will include robust error handling and detailed logging.

5. **Memory Management**: The scripts will include the memory monitoring and optimization from the notebooks.

6. **Parallel Processing**: Where appropriate, the scripts will use parallel processing to speed up execution.

## Usage Examples

### Training Data
```bash
# Process all training data
./extract_train_features.sh

# Process a specific target
./extract_train_features.sh --target "R1107"

# Process the first 5 targets (for testing)
./extract_train_features.sh --limit 5 --verbose
```

### Validation Data
```bash
# Process all validation data
./extract_validation_features.sh

# Process a specific validation target
./extract_validation_features.sh --target "R1149"
```

### Test Data
```bash
# Process all test data
./extract_test_features.sh

# Process with custom parameters
./extract_test_features.sh --csv data/raw/custom_test.csv --pf-scale 2.0
```

## Benefits of This Approach

1. **Full Feature Parity**: The shell scripts will produce identical outputs to the notebooks.

2. **Integration**: All feature types are handled in a single script for each data type.

3. **Environment Consistency**: The scripts ensure the correct environment is activated.

4. **Proper ViennaRNA Integration**: The ViennaRNA import issue will be resolved.

5. **Validation-Specific Handling**: The validation script includes the specialized handling for multiple coordinate sets.

6. **CLI Flexibility**: The scripts provide command-line flexibility while maintaining notebook functionality.

## Development Timeline

1. **Phase 1**: Implement the train features script and test with a small subset of data
2. **Phase 2**: Implement the validation features script with multi-coordinate handling
3. **Phase 3**: Implement the test features script
4. **Phase 4**: Comprehensive testing and verification
5. **Phase 5**: Integration with existing workflows and documentation updates

## Conclusion

This implementation plan provides a roadmap for creating reliable shell-based feature extraction scripts that match the functionality of the Jupyter notebooks. By directly translating the notebook code into standalone Python scripts called from shell wrappers, we can ensure full feature parity while fixing the issues in the current shell scripts.