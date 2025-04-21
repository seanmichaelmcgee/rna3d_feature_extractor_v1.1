# RNA 3D Feature Extractor Tests

This directory contains tests for the RNA 3D Feature Extractor.

## Test Structure

The tests are organized to mirror the project structure:

- `tests/data/` - Tests for data loading and management
- `tests/features/` - Tests for feature extraction
- `tests/processing/` - Tests for batch processing
- `tests/analysis/` - Tests for analysis utilities
- `tests/validation/` - Tests for feature validation
- `tests/integration/` - Integration tests for the entire workflow

## Running Tests

### Running All Tests

```bash
python -m unittest discover
```

### Running a Specific Test Module

```bash
python -m unittest tests.data.test_data_manager
```

### Running a Specific Test Case

```bash
python -m unittest tests.data.test_data_manager.TestDataManager.test_load_rna_data
```

## Test Requirements

Before running tests, ensure you have activated the RNA 3D feature extractor environment:

```bash
mamba activate rna3d-core
```

## Test Coverage

We aim for high test coverage, especially for core functionality. The following components are fully tested:

- DataManager
- FeatureExtractor
- BatchProcessor
- MemoryMonitor
- ResultValidator

## Integration Tests

Integration tests verify that all components work together correctly. These tests use mock data to simulate the RNA feature extraction pipeline.

## Adding New Tests

When adding new functionality, please also add appropriate tests. Follow these guidelines:

1. Create test files that mirror the module structure
2. Use descriptive test method names (`test_load_rna_data` instead of `test_1`)
3. Include both positive and negative test cases
4. Use `setUp` and `tearDown` methods for test fixtures
5. Use mocks for external dependencies

## Mock Data

The test suite uses mock data to avoid dependencies on large RNA datasets. Mock data includes:

- Small RNA sequences
- Simplified MSA data
- Pre-computed feature matrices

## Test Environment Variables

Some tests may use environment variables to configure behavior. Important variables:

- `RNA_TEST_DATA_DIR` - Override the test data directory
- `RNA_TEST_SKIP_SLOW` - Skip slow tests

## Legacy Tests

The original shell script-based tests are also available:

```bash
./tests/test_feature_extraction_scripts.sh
```

These tests verify:
1. Basic functionality of the extraction scripts
2. Resume capability for interrupted processes
3. Skip-existing functionality
4. Batch processing functionality
5. HTML report generation
6. Output feature format validity

After running the script tests, a comprehensive report will be generated at:
`tests/feature_extraction_test_report.md`