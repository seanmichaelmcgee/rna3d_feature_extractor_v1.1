# RNA Feature Extraction Tests

This directory contains tests for the RNA feature extraction shell scripts. The tests verify:

1. Basic functionality of the extraction scripts
2. Resume capability for interrupted processes
3. Skip-existing functionality
4. Batch processing functionality
5. HTML report generation
6. Output feature format validity

## Prerequisites

Before running tests, ensure:

1. The environment is properly set up using `./setup.sh`
2. The `rna3d-core` environment is activated: `mamba activate rna3d-core`
3. All required dependencies are installed (especially ViennaRNA)

The scripts will automatically check for proper environment configuration, but it's more efficient to verify the environment beforehand.

## Running Tests

To run all tests:

```bash
# Make sure the environment is activated
mamba activate rna3d-core

# Run tests
./tests/test_feature_extraction_scripts.sh
```

The test script will:
1. Run all tests in sequence
2. Generate a detailed test report in Markdown format
3. Store test output in the `data/test_output` directory
4. Log all test details to `tests/feature_extraction_test.log`

## Individual Tests

Each test can also be run individually:

### Basic Feature Extraction

```bash
# For training data
./scripts/feature_extraction/extract_train_features.sh --target 1SCL_A --output-dir data/test_train --limit 1

# For validation data
./scripts/feature_extraction/extract_validation_features.sh --target R1107 --output-dir data/test_validation --limit 1

# For test data
./scripts/feature_extraction/extract_test_features.sh --target R1107 --output-dir data/test_test --limit 1
```

### Feature Verification

```bash
# Verify features in a directory
./scripts/feature_extraction/verify_features.py data/test_train

# Or verify specific feature files
./scripts/feature_extraction/verify_features.py . --thermo-file data/test_train/thermo_features/1SCL_A_thermo_features.npz
```

### Resume and Skip Testing

```bash
# Basic test script for these functionalities
./scripts/feature_extraction/test_extraction_scripts.sh
```

## Test Report

After running the main test script, a comprehensive test report will be generated at:
`tests/feature_extraction_test_report.md`

The report includes:
- Test environment details
- Each test's description, command, and result
- Test output (truncated for readability)
- Overall success/failure status
- Duration of each test

## Adding New Tests

To add new tests:
1. Modify `test_feature_extraction_scripts.sh` 
2. Add a new test section using the `run_test` function
3. Follow the existing pattern for consistent reporting

```bash
run_test "Test Name" \
    "command to run" \
    "Description of what this test verifies"
```