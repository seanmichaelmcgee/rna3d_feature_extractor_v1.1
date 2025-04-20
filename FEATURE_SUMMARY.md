# RNA 3D Features Summary

## Feature Dataset Overview

This document provides a comprehensive summary of the RNA 3D feature dataset in the `data/processed` directory.

## Feature Count Statistics

- **Total unique RNA sequences**: 856
- **Feature Type Distribution**:
  - Dihedral features: 855 sequences
  - MI (Mutual Information) features: 856 sequences
  - Thermodynamic features: 856 sequences
- **Complete feature coverage**: 855 sequences (99.9%) have all three feature types extracted

## Feature Availability Analysis

- Only 1 sequence (`2OM3_R`) is missing dihedral features
- No sequences are missing MI features
- No sequences are missing thermodynamic features
- 855 sequences (99.9%) have all three feature types

## Sequence Naming Patterns

- **PDB-style IDs** (e.g., `1A60_A`): 844 sequences (98.6%)
  - Standard Protein Data Bank naming with 4-character PDB ID and chain identifier
- **R-style IDs** (e.g., `R1107`): 12 sequences (1.4%)
  - Internal research sequence identifiers
- **Versioned sequences**: 1 sequence (`R1117v2`)
  - Indicates version tracking for some sequences

## Feature Data Structure

### MI Features
- MI features directory (564MB) contains the majority of the feature data
- Special case: `7LHD_A_mi_features` subdirectory (136MB) with extended MI analysis:
  - `coupling_matrix.npy`
  - `method.npy`
  - `score_distance_correlation.npy`
  - `top_pairs.npy`

### Feature File Size Distribution
- Dihedral feature files: ~2.56 KB per file
- MI feature files: ~13.15 KB per file
- Thermodynamic feature files: ~16.98 KB per file

## Special Observations

1. **Consolidation**: Several MI and thermodynamic feature files for sequence R1107 with numbered suffixes (e.g., `R1107_10_mi_features.npz`) have been removed and consolidated.

2. **Storage Distribution**: MI features require significantly more storage (564MB) compared to dihedral (7.1MB) and thermodynamic features (46MB), indicating more complex data representation.

3. **Extended Analysis**: The `7LHD_A_mi_features` directory contains detailed mutual information analysis data (136MB), suggesting it may be a reference or template structure.

4. **Near-Perfect Coverage**: The 99.9% feature completeness indicates a well-maintained and consistent feature extraction pipeline.

## Data Integrity

The RNA feature dataset shows exceptional completeness and consistency across all feature types. The standardized naming conventions and high feature coverage rate indicate a robust feature extraction process suitable for machine learning applications.