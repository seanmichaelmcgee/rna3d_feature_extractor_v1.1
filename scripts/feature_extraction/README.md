# RNA Feature Extraction Shell Scripts

These shell scripts provide command-line alternatives to the Jupyter notebooks for extracting features from RNA sequences. The scripts support all three types of data (train, validation, test) and include robust error handling, progress tracking, and memory management.

## Prerequisites

- Conda/Mamba environment created with `environment.yml`
- ViennaRNA installed and properly configured
- Input data in the appropriate directories (`data/raw`)

## Usage

### Common Command-Line Options

All three scripts support the following options:

```
--csv FILE          CSV file with sequences (default: data/raw/[DATA_TYPE]_sequences.csv)
--output-dir DIR    Output directory (default: data/processed)
--limit N           Limit processing to N targets (for testing)
--cores N           Number of cores to use (default: all - 1)
--pf-scale SCALE    Partition function scaling factor (default: 1.5)
--target ID         Process only the specified target
--targets FILE      Process targets listed in file (one per line)
--skip-existing     Skip targets with existing feature files
--resume            Resume from last successful target
--batch-size N      Number of targets to process before cleanup (default: 5)
--memory-limit GB   Maximum memory usage in GB (default: 80% of system memory)
--verbose           Enable verbose output
--report            Generate HTML report after processing
--force             Overwrite existing output files
--help              Show help message
```

### Training Data

Extract features for training data with 3D structures:

```bash
./extract_train_features.sh
```

This extracts all three feature types:
- Thermodynamic features from RNA sequences
- Pseudodihedral angle features from 3D coordinates
- Mutual Information features from Multiple Sequence Alignments (MSAs)

### Validation Data

Process validation data with multiple coordinate sets:

```bash
./extract_validation_features.sh
```

Special handling is included for validation data:
- Multiple coordinate set processing
- Filtering of invalid coordinates (-1e+18 placeholder values)
- Flexible ID matching between different files

### Test Data

Process test data (which does not include 3D structures):

```bash
./extract_test_features.sh
```

For test data, only two feature types are extracted:
- Thermodynamic features from RNA sequences
- Mutual Information features from Multiple Sequence Alignments (MSAs)

## Examples

### Process Single Target

```bash
./extract_train_features.sh --target R1107 --verbose
```

### Process Limited Subset

```bash
./extract_validation_features.sh --limit 5 --report
```

### Resume Interrupted Processing

```bash
./extract_test_features.sh --resume --memory-limit 16
```

### Process Targets from File

```bash
./extract_train_features.sh --targets my_targets.txt --skip-existing
```

## Output

Each script generates the following outputs:

1. Feature files in NPZ format in the output directory
   - `thermo_features/[TARGET_ID]_thermo_features.npz`
   - `mi_features/[TARGET_ID]_mi_features.npz`
   - `dihedral_features/[TARGET_ID]_dihedral_features.npz` (train/validation only)

2. Progress tracking files
   - `logs/[DATA_TYPE]_progress.json`

3. Log files
   - `logs/[DATA_TYPE]_extraction_[TIMESTAMP].log`

4. Optional HTML reports (when using `--report`)
   - `reports/[DATA_TYPE]_report.html`

## Limitations and Troubleshooting

- For large RNA sequences (>1000 nt), consider using smaller batch sizes and higher memory limits
- If ViennaRNA errors occur, try adjusting the `--pf-scale` parameter
- For validation data with multiple coordinate sets, problems in one set will not affect the others
- The scripts provide detailed error messages in the log files
- Memory usage is monitored and the process will attempt cleanup if approaching limits
- For detailed error logging, use the `--verbose` flag

## Implementation Notes

These scripts were implemented based on the functionality in the Jupyter notebooks:
- `train_features_extraction.ipynb`
- `validation_features_extraction.ipynb`
- `test_features_extraction.ipynb`

They are designed to be more robust in production environments with extended error handling and progress tracking.