# RNA Feature Extraction: Comprehensive Testing Plan

## 1. Setup and Environment Preparation

### Mamba Environment Activation
```bash
# First activate Mamba shell
source ~/mambaforge/etc/profile.d/conda.sh
# Then activate the project environment
mamba activate rna3d-core
# Verify environment activation
python -c "import RNA; print(f'ViennaRNA version: {RNA.__version__}')"
```

### Data Verification
- Verify raw data directories contain expected files
- Confirm existence of train/validation/test files
- Check MSA files in data/raw/MSA/ directory
- Verify target data completeness (3D structure, sequence, MSA)

### Output Directory Preparation
- Ensure data/processed/ subdirectories exist for feature outputs
- Create consistent directory structure across notebooks

### ViennaRNA Version Check
- Verify ViennaRNA installation and version compatibility
- Confirm version 2.6.4 as specified in environment.yml
- Check proper initialization message from thermodynamic_analysis.py

## 2. File Naming and Output Structure Verification

### Output Consistency Check
- Verify each notebook produces consistently named outputs
- Ensure file naming follows pattern: `{target_id}_{feature_type}_features.npz`
- Confirm directory structure matches feature type:
  ```
  data/processed/thermo_features/{target_id}_thermo_features.npz
  data/processed/dihedral_features/{target_id}_dihedral_features.npz
  data/processed/mi_features/{target_id}_mi_features.npz
  ```
- Check that summary JSON files contain consistent metadata fields

### Cross-Notebook Compatibility
- Test loading features from one notebook into another
- Verify key feature names and shapes are consistent across notebooks
- Test npz_to_csv.py utility with outputs from all notebooks

## 3. Feature-Specific Testing

### Thermodynamic Features Validation
- Verify thermodynamic consistency check implementation
- Run explicit tests with `--validate-thermo` flag
- Check constraints: ensemble energy ≥ MFE
- Verify proper scaling with pf_scale parameter for longer sequences
- Test BPP matrix visualization for pattern validation

### MSA Quality Assessment
- Measure alignment depth (number of sequences)
- Calculate sequence diversity metrics
- Check for critical gaps in alignments
- Verify conservation pattern matches structural expectations
- Log statistics in JSON format for quality review

### Dihedral Features Verification
- Validate coordinate parsing from different file formats
- Verify proper handling of input structural coordinates
- Test pseudo-dihedral angle calculation with known structures
- Compare with reference structures from PDB

### Mutual Information Feature Validation
- Verify column-wise conservation and covariation
- Test correlation with known structural contacts
- Check for statistical significance of top pairs
- Validate performance with varying MSA depths (few vs. many sequences)
- Test chunking functionality for sequences > 750 nucleotides
- Verify MSA quality assessment and sequence weighting
- Test memory optimization with large MSAs
- Validate different configuration profiles (limited_resources, standard_workstation, high_performance)
- Test proper recombination of MI matrices from overlapping chunks
- Verify RNA-specific APC correction enhances signal-to-noise ratio

## 4. CLI Script Testing

### Script Feature Extraction
- Test individual script execution:
  ```
  python scripts/run_feature_extraction_single.sh target_id
  ```
- Test batch processing with run_mi_batch.sh
- Verify dihedral extraction with extract_pseudodihedral_features.py
- Test different flag combinations (e.g., --verbose, --pf-scale)

### CLI Parameter Testing
- Test with varying batch sizes
- Test with different output formats
- Test error handling with invalid inputs
- Verify proper logging and progress tracking

### Pipeline Integration
- Test end-to-end workflow with all scripts
- Verify proper error handling and output organization
- Test resuming functionality with checkpoint files

## 5. Testing Approach - Incremental Validation

### Initial Limited Dataset Testing
1. Set LIMIT=2 in each notebook
2. Run notebooks in sequence with a small, well-understood dataset
3. Verify all features are extracted correctly

### Feature Type Validation
1. **Thermodynamic Features:**
   - Verify MFE, structure, entropy values
   - Check pairing matrices for structural consistency
   - Visualize RNA structure and positional entropy

2. **Dihedral Features:**
   - Verify proper pseudodihedral angle calculation
   - Validate sine/cosine transformations
   - Verify 3D coordinate loading from different file formats

3. **Mutual Information Features:**
   - Verify MSA loading and alignment quality
   - Check MI matrix dimensions match sequence length
   - Validate top pairs with known structural contacts
   - Cross-reference with 3D distances (validation data only)
   - Test chunking functionality for sequences > 750 nucleotides
   - Verify sequence weighting reduces redundancy bias
   - Test RNA-specific APC correction implementation
   - Validate parameter optimization for different RNA lengths
   - Test memory usage monitoring with large MSAs
   - Compare enhanced MI output with basic implementation

### Cross-Dataset Validation
- Compare feature distributions between train and validation sets
- Ensure feature dimensions and structure are consistent
- Verify compatibility with the test dataset

## 6. Error Handling and Edge Cases

Test error handling with problematic inputs:
- Very short sequences (<10 nucleotides)
- Very long sequences (>2000 nucleotides)
- Targets with limited MSA data
- Targets with unusual structural properties
- Missing target files
- Incorrect file formats

## 7. Execution Plan

1. **Environment Setup (10 minutes)**
   - Activate mamba environment
   - Verify ViennaRNA installation
   - Check directory structure

2. **Single Target Testing (20 minutes)**
   - Select one well-understood target
   - Run through all three notebooks with LIMIT=1
   - Validate all output files and naming patterns
   - Check feature consistency and quality

3. **CLI Script Testing (20 minutes)**
   - Test each script with a single target
   - Verify outputs match notebook-generated files
   - Test batch processing with 3-5 targets

4. **Small Batch Testing (30 minutes)**
   - Run notebooks with LIMIT=5
   - Verify feature consistency across all targets
   - Check cross-validation metrics
   - Test with validation and test data
   - Test enhanced MI pipeline with different configuration profiles

5. **Edge Case Testing (30 minutes)**
   - Test with problematic inputs
   - Verify error handling
   - Check log messages and fallback mechanisms
   - Test very long RNA sequences (>2000 nt) with chunking
   - Test MSAs with varying quality and sequence count
   - Verify memory optimization for large datasets

6. **Complete Run (optional, time dependent)**
   - Run full datasets if time permits
   - Check resource usage and performance

7. **Documentation and Reporting (20 minutes)**
   - Document any issues found
   - Create summary statistics for feature quality
   - Generate reference visualizations of key features

## Testing Output Artifacts

For each test target, collect and organize:
1. NPZ files with extracted features
2. Feature quality metrics in JSON format
3. Visualization plots for key features
4. Log messages from execution
5. Memory and performance statistics