# Streamlined RNA Feature Extraction Testing Plan for Kaggle

## Overview

This testing plan focuses on ensuring our RNA feature extraction pipeline works reliably on Kaggle while setting the foundation for future scalability. The emphasis is on verification of core functionality, resource efficiency, and compatibility with Kaggle's constraints.

## 1. Environment Setup and Validation

### Mamba Environment Testing
```bash
# Activate environment
mamba activate rna3d-core

# Verify ViennaRNA installation
python -c "import RNA; print(f'ViennaRNA version: {RNA.__version__}')"

# Verify other critical dependencies
python -c "import numpy; import pandas; print('Core dependencies available')"
```

### Kaggle Environment Compatibility
- Verify all dependencies are available in Kaggle notebooks
- Identify any packages that need to be installed via pip
- Create a setup cell for the notebook that installs missing dependencies

## 2. Core Functionality Testing

### Thermodynamic Feature Extraction
- Test with a small RNA sequence (50-100nt)
- Verify MFE calculation works correctly
- Confirm base-pair probability matrix is generated
- Check positional entropy calculation

### Dihedral Feature Extraction
- Test with a small RNA structure
- Verify proper coordinate parsing
- Confirm angle calculations are correct

### Mutual Information Feature Extraction
- Test with a small MSA file
- Verify calculation works with limited sequences
- Test chunk processing for sequences > 750nt
- Validate pseudocount implementation for sparse MSAs
- Test adaptive pseudocount selection with different MSA sizes
- Verify integration with sequence weighting and APC correction

## 3. Resource Management Testing

### Memory Usage Assessment
- Monitor memory usage during feature extraction
- Identify memory bottlenecks
- Implement memory optimization strategies for large sequences
```python
# Memory usage tracking
import psutil
def log_memory_usage(label="Current memory"):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)
    print(f"{label}: {memory_gb:.2f} GB")

# Use at key points in processing
log_memory_usage("Before MI calculation")
# ... feature calculation ...
log_memory_usage("After MI calculation")
```

### Runtime Performance
- Measure processing time for different sequence lengths
- Ensure extraction completes within Kaggle time limits
- Optimize critical path functions

```python
import time

def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"{func.__name__} completed in {elapsed:.2f} seconds")
    return result, elapsed

# Use for performance-critical functions
features, elapsed = time_function(extract_thermodynamic_features, sequence)
```

## 4. Error Handling Verification

### Input Data Edge Cases
- Test with sequences containing non-standard nucleotides
- Verify proper handling of missing/incomplete data
- Test with extremely short and long sequences

### Graceful Failure Modes
- Ensure informative error messages
- Verify cleanup of temporary files on errors
- Test recovery from partial processing

## 5. Kaggle-Specific Verification

### Notebook Integration
- Test feature extraction functions as part of a Kaggle notebook
- Verify proper cell execution order
- Ensure notebook runs without user intervention

### Output Reliability
- Verify output files are correctly saved in Kaggle environment
- Confirm feature formats match downstream model requirements
- Test loading extracted features back into models

### Resource Monitoring in Notebook
```python
# Add to notebook for tracking resources
from IPython.display import display, clear_output
import time
import psutil

def monitor_resources(interval=30, duration=None):
    start_time = time.time()
    try:
        while True:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024 * 1024 * 1024)
            
            clear_output(wait=True)
            print(f"Time elapsed: {time.time() - start_time:.1f}s")
            print(f"Memory usage: {memory_gb:.2f} GB")
            print(f"CPU usage: {process.cpu_percent(interval=0.1)}%")
            
            # Check if duration limit reached
            if duration and (time.time() - start_time) > duration:
                break
                
            time.sleep(interval)
    except KeyboardInterrupt:
        pass

# Use in notebook for monitoring
# monitor_thread = threading.Thread(target=monitor_resources)
# monitor_thread.daemon = True
# monitor_thread.start()
```

## 6. Mini End-to-End Test

### Test Dataset Processing
- Create a small representative dataset (5-10 sequences)
- Process through entire pipeline
- Verify all feature types are correctly generated

### Validation Against Known Values
- Compare feature outputs to expected values for well-studied RNAs
- Verify thermodynamic consistency (ensemble energy ≥ MFE)
- Check structural features match known patterns

## 7. Documentation for Kaggle

### Notebook Documentation
- Add detailed comments in notebook cells
- Include resource usage expectations
- Document any limitations or constraints

### Feature Documentation
- Document feature formats and dimensions
- Explain biological significance of features
- Provide visualization examples for verification

## Implementation Checklist

This simplified checklist focuses on getting a working Kaggle submission with proper testing:

- [x] Verify environment setup works in Kaggle
- [x] Test core feature extraction on small examples
- [x] Create memory monitoring and profiling tools
- [ ] Identify and optimize memory bottlenecks
- [x] Handle edge cases and errors gracefully
- [x] Test with representative mini dataset
- [ ] Document the notebook thoroughly
- [ ] Verify end-to-end processing in Kaggle environment
- [x] Create MI pseudocount implementation plan
- [x] Implement pseudocount correction for basic MI calculation
- [x] Implement pseudocount integration with enhanced MI pipeline
- [x] Test pseudocount effectiveness with sparse MSAs
- [x] Fix structure data loading from labels CSV files
- [x] Test structure data loading with real examples

### Progress (April 12, 2025)

1. **Environment setup**: Verified mamba environment with ViennaRNA 2.6.4 and other dependencies
2. **Testing infrastructure**: Created comprehensive testing tools:
   - Feature verification script to ensure data loader compatibility
   - Memory monitoring system with visualization capabilities
   - Single target testing for end-to-end validation
   - Docker comparison utility for cross-environment validation
3. **Documentation**: Updated notebooks with memory monitoring, created implementation guide
4. **MI Enhancement**: Implemented pseudocount corrections:
   - Added adaptive pseudocount selection based on MSA size
   - Integrated with both basic and enhanced MI calculation
   - Ensured compatibility with sequence weighting and APC correction
   - Created comprehensive test suite and demonstration notebook
5. **Structure Data Loading**: Fixed and tested structure loading:
   - Updated load_structure_data to correctly use labels CSV files
   - Removed broken fallback logic for non-existent coordinate files
   - Created test script to verify correct implementation
   - Successfully tested with target R1107

## Future Extension Points

While not immediately necessary for Kaggle submission, these are marked for future development:

- Create comprehensive golden test cases
- Implement CI/CD integration
- Develop randomized test data generation
- Create containerized testing environment
- Build regression testing framework


## 8. Data Loader Integration Testing

### Feature Name and Format Validation
- Verify all feature names match exactly what the data loader expects
- Confirm file naming follows the required pattern for each feature type:
{target_id}_dihedral_features.npz
{target_id}_thermo_features.npz
{target_id}_features.npz  (in mi_features directory)
- Ensure directory structure matches what's expected by the data loader:
features_dir/
├── dihedral_features/
├── thermo_features/
└── mi_features/

### Downstream Loading Test
- Implement a simple test that uses the actual data loading component
```python
from data_loader import load_precomputed_features

# Test with one of your extracted feature sets
features = load_precomputed_features("test_target_id", "path/to/features_dir")

# Verify all required keys are present
assert 'dihedral' in features
assert 'thermo' in features
assert 'evolutionary' in features

# Verify critical features are available
assert 'pairing_probs' in features['thermo']
assert 'features' in features['dihedral']
Shape Verification

Verify that feature dimensions match what the data loader expects
Confirm tensor shapes follow: (sequence_length, 4) for dihedral features
Validate matrix shapes are both (sequence_length, sequence_length) for pairing_probs and coupling_matrix
