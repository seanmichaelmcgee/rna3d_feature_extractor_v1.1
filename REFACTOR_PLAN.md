# RNA 3D Feature Extractor Refactoring Plan

This document outlines the comprehensive refactoring plan for the RNA 3D Feature Extractor, focusing specifically on the `test_features_extraction.ipynb` notebook and its dependencies.

## 1. Objectives

- Create a modular, maintainable architecture
- Implement clean separation of concerns
- Optimize for Kaggle compatibility
- Retain single-sequence MSA optimization
- Ensure robust error handling and memory management
- Maintain backward compatibility with existing feature formats

## 2. Architecture Overview

The refactored codebase will be organized into these primary modules:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                 │       │                 │       │                 │
│   DataManager   │◄─────►│FeatureExtractor │◄─────►│ BatchProcessor  │
│                 │       │                 │       │                 │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         │                         │                         │
         │                         ▼                         │
         │                ┌─────────────────┐                │
         └───────────────►│                 │◄───────────────┘
                          │ MemoryMonitor   │
                          │                 │
                          └────────┬────────┘
                                   │
                                   │
                                   ▼
                          ┌─────────────────┐
                          │                 │
                          │ResultValidator  │
                          │                 │
                          └─────────────────┘
```

### 2.1 Module Structure

#### DataManager
Responsible for data I/O operations:
- Load RNA sequences from CSV files
- Load MSA data from FASTA files
- Save extracted features in NPZ format
- Support different data sources and paths
- Handle error recovery during data loading

#### FeatureExtractor
Handles feature calculation:
- Thermodynamic feature extraction using ViennaRNA
- Mutual Information calculation with single-sequence optimization
- Parameter configuration for feature extraction
- Feature normalization and formatting

#### BatchProcessor
Manages the processing workflow:
- Target selection and filtering
- Batch size optimization
- Memory-aware scheduling
- Parallel processing when appropriate
- Progress tracking and resumption

#### MemoryMonitor
Tracks and optimizes resource usage:
- Memory usage monitoring
- Resource allocation strategies
- Adaptive batch sizing
- Peak memory reporting

#### ResultValidator
Validates feature compatibility:
- Format validation
- Dimension checking
- Content verification
- Compatibility with downstream models

## 3. Implementation Plan

### 3.1 Core Refactoring Steps

1. **Extract Common Functions**
   - Move helper functions from notebook to appropriate modules
   - Standardize function signatures and return types
   - Add type hints and comprehensive docstrings

2. **Create Class Hierarchy**
   - Implement DataManager class
   - Implement FeatureExtractor class
   - Implement BatchProcessor class
   - Implement MemoryMonitor class
   - Implement ResultValidator class

3. **Implement Configuration System**
   - Create configuration management approach
   - Support both file-based and programmatic configuration
   - Include sensible defaults for all parameters

4. **Optimize Memory Management**
   - Implement memory-aware batch processing
   - Add cleanup routines for large objects
   - Implement configurable resource limits

5. **Enhance Error Handling**
   - Add robust exception handling
   - Implement recovery mechanisms
   - Create detailed error reporting

6. **Maintain Notebook Interface**
   - Create simplified notebook that uses refactored modules
   - Ensure backward compatibility
   - Add examples for common use cases

### 3.2 Detailed Module Specifications

#### DataManager Class

```python
class DataManager:
    """
    Handles data loading, saving, and format conversion for RNA feature extraction.
    """
    
    def __init__(self, data_dir=None, raw_dir=None, processed_dir=None):
        """Initialize with configurable data directories."""
        
    def load_rna_data(self, csv_path):
        """Load RNA data from CSV file."""
        
    def get_unique_target_ids(self, df, id_col="ID"):
        """Extract unique target IDs from dataframe."""
        
    def load_msa_data(self, target_id, data_dir=None):
        """Load MSA data for a given target."""
        
    def get_sequence_for_target(self, target_id, data_dir=None):
        """Get RNA sequence for a target ID from the sequence file."""
        
    def save_features(self, features, output_file):
        """Save extracted features to NPZ file."""
        
    def load_features(self, target_id, feature_type):
        """Load features for a target ID."""
```

#### FeatureExtractor Class

```python
class FeatureExtractor:
    """
    Extracts various features from RNA sequences and alignments.
    """
    
    def __init__(self, memory_monitor=None, verbose=False):
        """Initialize with optional memory monitoring."""
        
    def extract_thermodynamic_features(self, sequence, pf_scale=1.5):
        """Extract thermodynamic features for an RNA sequence."""
        
    def extract_mi_features(self, msa_sequences, pseudocount=None):
        """Extract Mutual Information features from MSA sequences."""
        
    def validate_features(self, features, feature_type):
        """Validate extracted features for correctness and format compatibility."""
```

#### BatchProcessor Class

```python
class BatchProcessor:
    """
    Manages batch processing of multiple RNA targets.
    """
    
    def __init__(self, data_manager, feature_extractor, 
                 memory_monitor=None, batch_size=5, verbose=False):
        """Initialize with required components and settings."""
        
    def process_target(self, target_id, extract_thermo=True, extract_mi=True):
        """Process a single target, extracting requested feature types."""
        
    def batch_process_targets(self, target_ids, extract_thermo=True, extract_mi=True):
        """Process multiple targets in batch mode."""
        
    def get_optimal_batch_size(self, target_ids, available_memory):
        """Determine optimal batch size based on targets and available memory."""
```

#### MemoryMonitor Class

```python
class MemoryMonitor:
    """
    Monitors and manages memory usage during feature extraction.
    """
    
    def __init__(self, memory_limit=None, verbose=False):
        """Initialize with optional memory limit."""
        
    def start_tracking(self, label):
        """Start tracking memory usage for a specific operation."""
        
    def stop_tracking(self):
        """Stop tracking and return memory usage statistics."""
        
    def log_memory_usage(self, label):
        """Log current memory usage with a descriptive label."""
        
    def plot_memory_usage(self, output_file=None):
        """Generate plot of memory usage over time."""
        
    def check_memory_limit(self):
        """Check if current memory usage is approaching the limit."""
```

#### ResultValidator Class

```python
class ResultValidator:
    """
    Validates feature extraction results for consistency and compatibility.
    """
    
    def __init__(self, verbose=False):
        """Initialize validator with verbosity setting."""
        
    def validate_thermodynamic_features(self, features):
        """Validate thermodynamic features."""
        
    def validate_mi_features(self, features):
        """Validate mutual information features."""
        
    def validate_feature_compatibility(self, features):
        """Validate compatibility with downstream processing."""
        
    def generate_validation_report(self, validation_results):
        """Generate a validation report based on validation results."""
```

## 4. Migration Strategy

The migration from the current notebook to the refactored modules will follow these steps:

1. **Create Module Framework**
   - Set up directory structure
   - Create class skeletons
   - Establish interfaces between components

2. **Migrate Core Functionality**
   - Move data loading functions to DataManager
   - Move feature extraction functions to FeatureExtractor
   - Move batch processing functions to BatchProcessor
   - Move memory monitoring to MemoryMonitor
   - Move validation to ResultValidator

3. **Create Integration Tests**
   - Develop tests for each module
   - Create integration tests for common workflows
   - Validate feature compatibility with current outputs

4. **Create Simplified Notebook**
   - Implement new version of notebook using refactored modules
   - Ensure visual outputs match current notebook
   - Add additional documentation for new capabilities

5. **Validate End-to-End**
   - Process test dataset using refactored code
   - Compare outputs with previous implementation
   - Verify memory usage and performance improvements

## 5. Directory Structure

The refactored codebase will use this directory structure:

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── data_manager.py
├── features/
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── thermodynamic_features.py
│   └── mutual_information_features.py
├── processing/
│   ├── __init__.py
│   ├── batch_processor.py
│   └── memory_monitor.py
├── validation/
│   ├── __init__.py
│   └── result_validator.py
└── utils/
    ├── __init__.py
    └── configuration.py
```

## 6. Backward Compatibility

To ensure backward compatibility:

1. **Feature Format Consistency**: Maintain the same NPZ structure with identical key names
2. **Function Signatures**: Keep core functions with similar signatures for dependent code
3. **Notebook Interface**: Provide notebook with identical inputs/outputs to current version
4. **Error Handling**: Be more robust but follow similar error reporting patterns

## 7. Performance Considerations

The refactored code will prioritize these performance aspects:

1. **Memory Efficiency**: Optimize for low memory footprint, especially for large RNAs
2. **Batch Processing**: Improve batch processing with better memory management
3. **Single-Sequence MSA**: Maintain the optimization for single-sequence MSAs
4. **Parallelization**: Add controlled parallelization where appropriate

## 8. Testing Strategy

The refactoring will include:

1. **Unit Tests**: For individual components
2. **Integration Tests**: For module interactions
3. **End-to-End Tests**: For full workflows
4. **Memory Tests**: For resource usage validation
5. **Edge Case Tests**: For unusual inputs and error handling

## 9. Documentation

Documentation will be enhanced with:

1. **Module Documentation**: Complete docstrings for all classes and methods
2. **Usage Examples**: Example code for common use cases
3. **Architecture Overview**: Documentation of component interactions
4. **Configuration Guide**: Documentation of all configurable parameters
5. **Troubleshooting Guide**: Common issues and solutions

## 10. Future Extensibility

The architecture is designed for future extensions:

1. **New Feature Types**: Easy addition of new feature extractors
2. **Alternative Backends**: Support for different computational backends
3. **Visualization Enhancements**: Pluggable visualization system
4. **Pipeline Integration**: Hooks for integration with ML pipelines