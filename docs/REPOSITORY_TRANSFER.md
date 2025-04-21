# Repository Transfer Guide

This document provides guidance on how to transfer the refactored codebase to a new repository.

## Prerequisites

Before transferring the repository, ensure:

1. All tests are passing
2. Documentation is complete
3. All features are implemented and working correctly
4. The codebase is in a clean state (no uncommitted changes)

## Transfer Methods

There are two main methods for transferring the repository:

### Method 1: Clone and Push

This method preserves the commit history.

```bash
# Clone the current repository
git clone /path/to/current/repo new-repo
cd new-repo

# Change the remote to the new repository
git remote remove origin
git remote add origin https://github.com/new-owner/new-repo.git

# Push to the new repository
git push -u origin main
```

### Method 2: Archive and Initialize

This method starts with a clean history.

```bash
# Create a new repository
git init new-repo
cd new-repo

# Copy everything from the old repository (excluding .git directory)
cp -r /path/to/current/repo/* .
cp -r /path/to/current/repo/.github .  # If .github directory exists

# Commit all files
git add .
git commit -m "Initial commit of refactored codebase"

# Add the remote and push
git remote add origin https://github.com/new-owner/new-repo.git
git push -u origin main
```

## Verification Steps

After transferring the repository, verify:

1. **Files**: All files are present and correctly structured
2. **Dependencies**: All dependencies are correctly specified
3. **Tests**: All tests run and pass in the new repository
4. **Documentation**: All documentation links work correctly
5. **Execution**: The code runs as expected in the new environment

## Repository Structure Overview

The repository has the following structure:

```
rna3d_feature_extractor/
├── data/                      # Data directory
│   ├── processed/             # Processed data
│   └── raw/                   # Raw data
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks
├── scripts/                   # Scripts for running feature extraction
├── src/                       # Source code
│   ├── data/                  # Data loading and management
│   ├── features/              # Feature extraction
│   ├── processing/            # Batch processing
│   ├── analysis/              # Analysis utilities
│   └── validation/            # Feature validation
├── tests/                     # Tests
│   ├── data/                  # Tests for data module
│   ├── features/              # Tests for features module
│   ├── processing/            # Tests for processing module
│   ├── analysis/              # Tests for analysis module
│   └── validation/            # Tests for validation module
├── environment.yml            # Conda environment definition
├── LICENSE                    # License file
├── README.md                  # Main README
├── setup.py                   # Package setup file
└── setup.sh                   # Environment setup script
```

## Post-Transfer Tasks

After transferring the repository:

1. **Update Documentation**: Update any links that referenced the old repository
2. **CI/CD**: Set up continuous integration and deployment
3. **Dependencies**: Verify that all dependencies are correctly specified
4. **Permissions**: Set appropriate permissions for the new repository
5. **Issues**: Transfer any open issues to the new repository

## Repository Health Check

Before concluding the transfer, run the following health check:

```bash
# Activate environment
mamba activate rna3d-core

# Run all tests
python -m unittest discover

# Verify installation
pip install -e .

# Run a simple extraction to verify functionality
python scripts/run_feature_extraction_single.sh sample_target
```

## Troubleshooting

If you encounter issues after the transfer:

1. **Path Issues**: Check for hardcoded paths that might need to be updated
2. **Permission Issues**: Ensure all files have appropriate permissions
3. **Package Issues**: Verify that the package is correctly installed
4. **Environment Issues**: Check that all dependencies are correctly specified

## Contact

If you encounter any issues with the repository transfer, please contact the original developers.