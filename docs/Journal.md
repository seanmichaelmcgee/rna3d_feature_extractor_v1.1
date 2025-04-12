# Testing Journal

This document tracks the progress of testing the RNA feature extraction pipeline according to our streamlined testing plan focused on Kaggle submission.

## Updated Testing Goals

We've revised our testing strategy to focus on:
1. Getting the feature extraction pipeline working for Kaggle submissions
2. Ensuring compatibility with the downstream PyTorch data loading component
3. Setting a foundation for future potential scalability
4. Using Docker to verify dependency management and environment consistency

## Execution Plan Progress

### 1. Environment Setup (30 minutes)
- [x] Activate mamba environment
- [x] Verify ViennaRNA installation
- [x] Check directory structure and permissions
- [x] Update testing plans for streamlined goals
- [x] Create verification script for data loader compatibility
- [x] **Docker Verification**: Build and test Docker image to ensure all dependencies are correctly installed
  ```bash
  docker build -t rna3d-extractor .
  docker run --rm rna3d-extractor micromamba run -n rna3d-core python -c "import RNA; print(f'ViennaRNA version: {RNA.__version__}')"
  ```

### 2. Single Target Testing (40 minutes)
- [x] Create script for single target testing (`scripts/single_target_test.py`)
- [ ] Select one well-understood target
- [ ] Run through all three notebooks with LIMIT=1
- [ ] Validate all output files and naming patterns
- [ ] Check feature consistency and quality
- [ ] Verify compatibility with data loader requirements
- [x] Create tool for Docker output comparison (`scripts/compare_docker_outputs.py`)
- [ ] **Docker Verification**: Process the same target in Docker to compare outputs
  ```bash
  # Run same test in Docker environment
  docker run --rm \
    -v $(pwd)/data/raw:/app/data/raw \
    -v $(pwd)/data/processed:/app/data/processed \
    --entrypoint /bin/bash \
    -c "micromamba run -n rna3d-core python scripts/single_target_test.py --target R1107" \
    rna3d-extractor
  
  # Compare outputs between local and Docker runs
  python scripts/compare_docker_outputs.py data/processed data/processed_docker
  ```

### 3. Feature Format Validation (30 minutes)
- [x] Create feature verification script (`scripts/verify_feature_compatibility.py`)
- [ ] Verify file naming matches data loader expectations
- [ ] Confirm directory structure is correct
- [ ] Test feature loading with data loader functions
- [ ] Verify tensor shapes and types match requirements
- [ ] **Docker Verification**: Run feature validation script in Docker to ensure format consistency
  ```bash
  docker run --rm \
    -v $(pwd)/data/processed:/app/data/processed \
    --entrypoint /bin/bash \
    -c "micromamba run -n rna3d-core python scripts/verify_feature_compatibility.py /app/data/processed" \
    rna3d-extractor
  ```

### 4. Resource Management (30 minutes)
- [x] Add memory monitoring to notebooks
- [x] Create tools for profiling different RNA lengths
- [ ] Identify potential memory bottlenecks
- [ ] Optimize for Kaggle's memory constraints
- [ ] **Docker Verification**: Measure resource usage in constrained Docker environment
  ```bash
  # Test with memory constraints similar to Kaggle
  docker run --rm \
    -v $(pwd)/data/raw:/app/data/raw \
    -v $(pwd)/data/processed:/app/data/processed \
    --memory=8g --cpus=2 \
    --entrypoint /bin/bash \
    -c "micromamba run -n rna3d-core python scripts/single_target_test.py --target R1107" \
    rna3d-extractor
  ```

### 5. Mini End-to-End Test (40 minutes)
- [ ] Process a small representative dataset
- [ ] Verify all feature types are correctly generated
- [ ] Test loading features with the data loading component
- [ ] Validate output tensor shapes and types
- [ ] **Docker Verification**: Run end-to-end test in Docker
  ```bash
  # Process mini dataset in Docker
  docker run --rm \
    -v $(pwd)/data/raw:/app/data/raw \
    -v $(pwd)/data/processed:/app/data/processed \
    --entrypoint /bin/bash \
    -c "micromamba run -n rna3d-core python src/data/batch_feature_runner.py --csv data/raw/mini_dataset.csv" \
    rna3d-extractor
  
  # Verify data loader compatibility
  docker run --rm \
    -v $(pwd)/data/processed:/app/data/processed \
    --entrypoint /bin/bash \
    -c "micromamba run -n rna3d-core python scripts/test_data_loader.py" \
    rna3d-extractor
  ```

### 6. Documentation (20 minutes)
- [ ] Update notebooks with clear comments
- [ ] Document resource requirements and limitations
- [ ] Add verification steps to check output compatibility
- [ ] Provide instructions for Kaggle submission
- [ ] **Docker Documentation**: Add notes on using Docker for testing and development

## Detailed Testing Notes

### Step 1: Environment Setup
**Completed on:** April 10, 2025

**Environment Activation:**
- Successfully initialized mamba shell with: `eval "$(mamba shell hook --shell bash)"`
- Activated rna3d-core environment: `mamba activate rna3d-core`
- Verified ViennaRNA version: 2.6.4 (matches environment.yml specification)

**Directory Structure:**
- Created required directories: data/processed/{thermo_features,dihedral_features,mi_features}
- Verified raw data directory contains necessary MSA files
- Verified existence of train, validation, and test files

**Docker Environment Setup:**
- Created Dockerfile based on micromamba with our environment.yml
- Built Docker image: `docker build -t rna3d-extractor .`
- Verified dependencies in Docker environment:
  ```
  docker run --rm rna3d-extractor micromamba run -n rna3d-core python -c "import RNA; print(f'ViennaRNA version: {RNA.__version__}')"
  ```
- Confirmed all dependencies match between local and Docker environments

**Updated Testing Approach:**
- Reviewed downstream data loading component requirements
- Updated testing plans to focus on compatibility with PyTorch pipeline
- Created verification script for checking feature format compatibility

**Key Requirements Identified:**
- Feature file naming must follow specific patterns:
  - `{target_id}_dihedral_features.npz`
  - `{target_id}_thermo_features.npz`
  - `{target_id}_features.npz` (for MI features)
- Directory structure must match data loader expectations
- Feature keys must match exactly what the loader expects
- Shape requirements must be met for all tensor types

### Step A: Dockerfile Creation
**Completed on:** April 11, 2025

**Docker Environment:**
- Created streamlined Dockerfile with micromamba base
- Added environment setup from environment.yml
- Configured volume mounts for data directories
- Set up entrypoint for running feature extraction script
- Added directory structure creation for feature outputs:
  ```
  RUN mkdir -p /app/data/processed/dihedral_features \
               /app/data/processed/thermo_features \
               /app/data/processed/mi_features
  ```

**Docker Dependency Verification:**
- Built Docker image to confirm environment setup
- Verified key dependencies:
  ```
  docker run --rm rna3d-extractor micromamba run -n rna3d-core python -c "import RNA; import numpy; import pandas; print('Core dependencies verified')"
  ```
- Created test script to verify ViennaRNA functionality:
  ```python
  # test_vienna.py
  import RNA
  seq = "GGGAAACCC"
  (ss, mfe) = RNA.fold(seq)
  print(f"Sequence: {seq}")
  print(f"Structure: {ss}")
  print(f"MFE: {mfe}")
  
  # Run in Docker
  docker run --rm rna3d-extractor micromamba run -n rna3d-core python test_vienna.py
  ```

### Step 2: Single Target Testing
*(Notes will be added as testing progresses)*

### Step 3: Feature Format Validation
*(Notes will be added as testing progresses)*

### Step 4: Resource Management
**Initial implementation completed on:** April 12, 2025

**Memory Monitoring System:**
- Created comprehensive memory tracking module: `src/analysis/memory_monitor.py`
- Implemented flexible tracking mechanisms:
  - `log_memory_usage()` function for spot measurements
  - `MemoryTracker` context manager for tracking code sections
  - Decorator for monitoring specific functions
- Added visualization capabilities for memory usage over time
- Created profiling utility for different RNA sequence lengths

**Notebook Integration:**
- Updated training notebook with memory monitoring points
- Added memory usage visualization cell 
- Created RNA length profiling function to estimate memory requirements
- Set up infrastructure for bottleneck identification

**Next steps:**
- Run profiling with various RNA lengths to establish memory scaling
- Identify specific bottlenecks in the feature extraction process
- Implement optimization strategies for large sequences (>1000nt)
- Document memory requirements for Kaggle environment

### Step 5: Mini End-to-End Test
**Infrastructure implementation completed on:** April 12, 2025

**Single Target Testing:**
- Created comprehensive end-to-end testing script: `scripts/single_target_test.py`
- Implemented all three feature types extraction with memory tracking
- Added error handling and detailed logging
- Integrated feature validation steps

**Docker Comparison:**
- Created Docker output comparison utility: `scripts/compare_docker_outputs.py`
- Implemented file existence checks and content validation
- Set up detailed reporting of any differences

**Next steps:**
- Run actual tests with representative targets
- Process sequences of various lengths to test scaling
- Compare local and Docker outputs for consistency
- Document results in testing journal

### Step 6: Documentation
**Initial documentation added on:** April 12, 2025

**Implementation Summary:**
- Created summary document: `docs/implementation-summary.md`
- Documented all new tools and usage instructions
- Added detailed inline documentation to all scripts
- Updated Journal with progress and next steps

**Notebook Documentation:**
- Added memory monitoring cells
- Created profiling capabilities
- Added verification cell for data loader compatibility

**Next steps:**
- Add detailed memory requirements based on profiling
- Document optimization strategies for large sequences
- Update implementation summary with test results
- Create user guide for Kaggle submission

## Issues and Recommendations

### Docker Role in Testing

We're using Docker for several important purposes in our testing strategy:

1. **Clean dependency verification**:
   - Docker provides an isolated environment to verify all dependencies
   - This helps identify potential conflicts or missing packages
   - Particularly important for ViennaRNA which can be complex to install

2. **Environment consistency**:
   - Docker ensures identical testing environment across different machines
   - Helps identify issues that might be environment-specific
   - Provides a reference for how to set up Kaggle environment

3. **Feature format verification**:
   - Running validation in Docker confirms features are consistently generated
   - Helps ensure compatibility with the data loading component

4. **Resource constraints testing**:
   - Docker allows simulation of memory/CPU constraints similar to Kaggle
   - Helps identify potential performance issues in constrained environments

While Docker itself isn't used in Kaggle, it provides valuable verification that our pipeline works consistently across environments and helps ensure all dependencies are properly managed.

### Data Loader Compatibility

After reviewing the PyTorch data loading component, we've identified several critical requirements:

1. **File naming and locations**:
   - The data loader expects specific file names and directory structure
   - We need to ensure our pipeline produces outputs in the expected format
   - Directory structure must follow: features_dir/{dihedral_features,thermo_features,mi_features}

2. **Feature naming conventions**:
   - Key feature names must match exactly: 'pairing_probs', 'positional_entropy', 'features', 'coupling_matrix'
   - The loader does have some fallbacks (e.g., 'base_pair_probs' for 'pairing_probs')
   - We should standardize our feature names to match primary expectations

3. **Shape requirements**:
   - Dihedral features must be (sequence_length, 4)
   - Pairing probabilities and coupling matrix must be (sequence_length, sequence_length)
   - Positional entropy and accessibility must be (sequence_length,)

4. **Testing priority**:
   - We need to add explicit verification that our features can be loaded by the data loading component
   - Create a simple test script that uses the actual data loader functions
   - Verify all shapes and types match expectations

### Memory Optimization Strategy

For Kaggle submissions, memory optimization is critical:

1. **Monitoring approach**:
   - Add memory monitoring code to notebooks
   - Track peak memory usage during feature extraction
   - Identify memory bottlenecks in the pipeline

2. **Optimization techniques**:
   - Process sequences in smaller batches
   - Clear variables when no longer needed
   - Use chunking for MI calculations on long sequences
   - Consider lower precision (float16) for large matrices if memory is tight

3. **Documentation needs**:
   - Document memory requirements for different sequence lengths
   - Provide guidance on batch sizes for different RNA types
   - Include memory monitoring cells in notebooks

### Progress Update - April 12, 2025

We've made significant progress implementing key testing infrastructure:

1. **Completed tools**:
   - Created feature verification script (`scripts/verify_feature_compatibility.py`)
   - Added memory monitoring system (`src/analysis/memory_monitor.py`)
   - Created single target testing script (`scripts/single_target_test.py`)
   - Implemented Docker comparison utility (`scripts/compare_docker_outputs.py`)
   - Added memory profiling to notebooks with visualization
   - Created MI pseudocount implementation plan (`docs/MiPseudoCount.md`)

2. **Next actions**:
   - Run single target test and verify outputs
   - Process a mini dataset with different RNA lengths
   - Identify and optimize memory bottlenecks for large sequences
   - Document memory requirements for Kaggle
   - Compare feature extraction consistency between local and Docker environments
   - Implement pseudocount enhancement for mutual information calculations

3. **Summary documentation**:
   - Created implementation summary document with usage instructions
   - Updated notebooks with memory monitoring capabilities
   - Added detailed function documentation to all new scripts
   - Documented comprehensive MI pseudocount implementation approach
