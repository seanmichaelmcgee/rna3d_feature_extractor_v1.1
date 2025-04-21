# RNA 3D Feature Extractor Refactoring Component Index

This document provides a comprehensive index of all components in the refactored RNA 3D Feature Extractor codebase, with dependencies, responsibilities, and interaction patterns.

## 1. Component Dependency Map

The following diagram shows the primary dependencies between components:

```
DataManager ─────► FeatureExtractor ◄───── ResultValidator
    ▲                    ▲                       ▲
    │                    │                       │
    │                    │                       │
    └───────► BatchProcessor ◄──────────────────┘
                    ▲
                    │
                    │
               MemoryMonitor
```

## 2. Core Component Details

### 2.1 DataManager

**Location**: `src/data/data_manager.py`

**Primary Responsibilities**:
- Load RNA sequences from various formats
- Load MSA data from FASTA files
- Save extracted features
- Handle data validation and error recovery

**Key Dependencies**:
- Pandas (for CSV loading)
- NumPy (for feature array management)
- Path (for file path handling)

**Public Interface**:
- `load_rna_data(csv_path)` - Load RNA data from CSV
- `get_unique_target_ids(df, id_col)` - Extract unique target IDs
- `load_msa_data(target_id, data_dir)` - Load MSA data
- `get_sequence_for_target(target_id, data_dir)` - Get RNA sequence
- `save_features(features, output_file)` - Save extracted features
- `load_features(target_id, feature_type)` - Load previously extracted features

**Integration Points**:
- Called by BatchProcessor for data loading
- Provides data to FeatureExtractor
- Manages feature file I/O

### 2.2 FeatureExtractor

**Location**: `src/features/feature_extractor.py`

**Primary Responsibilities**:
- Extract thermodynamic features using ViennaRNA
- Calculate mutual information from MSA data
- Handle feature normalization and formatting
- Manage extraction parameters

**Key Dependencies**:
- ViennaRNA (for thermodynamic calculations)
- NumPy (for numerical operations)
- SciPy (for additional calculations)
- MemoryMonitor (for resource tracking)

**Public Interface**:
- `extract_thermodynamic_features(sequence, pf_scale)` - Extract thermo features
- `extract_mi_features(msa_sequences, pseudocount)` - Extract MI features
- `validate_features(features, feature_type)` - Validate feature format

**Integration Points**:
- Called by BatchProcessor for feature extraction
- Uses DataManager to access sequences and MSA data
- Monitored by MemoryMonitor
- Features validated by ResultValidator

**Modules**:
- `thermodynamic_features.py` - Implementation of thermodynamic feature extraction
- `mutual_information_features.py` - Implementation of MI feature extraction

### 2.3 BatchProcessor

**Location**: `src/processing/batch_processor.py`

**Primary Responsibilities**:
- Manage batch processing of multiple targets
- Coordinate data loading and feature extraction
- Handle memory-aware scheduling
- Track progress and results

**Key Dependencies**:
- DataManager (for data access)
- FeatureExtractor (for feature calculation)
- MemoryMonitor (for resource management)
- ResultValidator (for output validation)

**Public Interface**:
- `process_target(target_id, extract_thermo, extract_mi)` - Process single target
- `batch_process_targets(target_ids, extract_thermo, extract_mi)` - Process multiple targets
- `get_optimal_batch_size(target_ids, available_memory)` - Calculate optimal batch size

**Integration Points**:
- Central coordinator between all components
- Called by notebook or command line interface
- Provides results to result visualization

### 2.4 MemoryMonitor

**Location**: `src/processing/memory_monitor.py`

**Primary Responsibilities**:
- Track memory usage during processing
- Log memory statistics
- Generate memory usage visualizations
- Enforce memory limits

**Key Dependencies**:
- psutil (for memory monitoring)
- matplotlib (for visualization)
- gc (for garbage collection)

**Public Interface**:
- `start_tracking(label)` - Begin tracking for an operation
- `stop_tracking()` - End tracking and return statistics
- `log_memory_usage(label)` - Log current memory usage
- `plot_memory_usage(output_file)` - Generate memory usage plot
- `check_memory_limit()` - Check if approaching memory limit

**Integration Points**:
- Used by BatchProcessor for resource management
- Used by FeatureExtractor during intensive calculations
- Provides data for performance reports

### 2.5 ResultValidator

**Location**: `src/validation/result_validator.py`

**Primary Responsibilities**:
- Validate feature format compatibility
- Check feature dimensions and content
- Generate validation reports
- Ensure compatibility with downstream models

**Key Dependencies**:
- NumPy (for array validation)
- json (for report generation)

**Public Interface**:
- `validate_thermodynamic_features(features)` - Validate thermo features
- `validate_mi_features(features)` - Validate MI features
- `validate_feature_compatibility(features)` - Check compatibility
- `generate_validation_report(validation_results)` - Create validation report

**Integration Points**:
- Called by BatchProcessor after feature extraction
- Provides validation feedback to user
- Ensures proper format for downstream analysis

## 3. Utility Components

### 3.1 Configuration Module

**Location**: `src/utils/configuration.py`

**Primary Responsibilities**:
- Manage configuration parameters
- Load configuration from files
- Provide defaults for missing parameters
- Support environment-specific configuration

**Key Dependencies**:
- json (for config file parsing)
- os (for environment variable access)

**Public Interface**:
- `load_config(config_file)` - Load configuration from file
- `get_config_value(key, default)` - Get configuration parameter
- `save_config(config, config_file)` - Save configuration
- `detect_environment()` - Detect and configure for environment

### 3.2 Visualization Module

**Location**: `src/utils/visualization.py`

**Primary Responsibilities**:
- Generate visualizations of features
- Create RNA structure diagrams
- Plot feature matrices and statistics
- Generate reports with visualizations

**Key Dependencies**:
- matplotlib (for plotting)
- seaborn (for enhanced visualizations)
- pandas (for data manipulation)

**Public Interface**:
- `visualize_rna_structure(sequence, structure)` - Show RNA structure
- `plot_mi_matrix(mi_features)` - Plot MI matrix
- `plot_thermodynamic_features(thermo_features)` - Visualize thermo features
- `generate_feature_report(features, output_file)` - Create feature report

## 4. Dependency Mapping for Original Files

This section maps functions from the original notebook to their new locations in the refactored architecture.

### 4.1 Data Loading Functions

| Original Function | New Location | New Function/Method |
|-------------------|--------------|---------------------|
| `load_rna_data` | `src/data/data_manager.py` | `DataManager.load_rna_data` |
| `get_unique_target_ids` | `src/data/data_manager.py` | `DataManager.get_unique_target_ids` |
| `load_msa_data` | `src/data/data_manager.py` | `DataManager.load_msa_data` |
| `get_sequence_for_target` | `src/data/data_manager.py` | `DataManager.get_sequence_for_target` |

### 4.2 Feature Extraction Functions

| Original Function | New Location | New Function/Method |
|-------------------|--------------|---------------------|
| `extract_thermodynamic_features_for_target` | `src/features/feature_extractor.py` | `FeatureExtractor.extract_thermodynamic_features` |
| `extract_thermodynamic_features` (from src/analysis) | `src/features/thermodynamic_features.py` | `extract_thermodynamic_features` |
| `extract_mi_features_for_target` | `src/features/feature_extractor.py` | `FeatureExtractor.extract_mi_features` |
| `calculate_mutual_information` (from src/analysis) | `src/features/mutual_information_features.py` | `calculate_mutual_information` |

### 4.3 Batch Processing Functions

| Original Function | New Location | New Function/Method |
|-------------------|--------------|---------------------|
| `process_target` | `src/processing/batch_processor.py` | `BatchProcessor.process_target` |
| `batch_process_targets` | `src/processing/batch_processor.py` | `BatchProcessor.batch_process_targets` |

### 4.4 Memory Monitoring Functions

| Original Function | New Location | New Function/Method |
|-------------------|--------------|---------------------|
| `MemoryTracker` class | `src/processing/memory_monitor.py` | `MemoryMonitor` class |
| `log_memory_usage` | `src/processing/memory_monitor.py` | `MemoryMonitor.log_memory_usage` |
| `plot_memory_usage` | `src/processing/memory_monitor.py` | `MemoryMonitor.plot_memory_usage` |

## 5. Import Changes

This section shows how imports will change in the refactored codebase:

### 5.1 Original Imports

```python
# Standard imports
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
import json
import psutil

# Ensure the parent directory is in the path so we can import our modules
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import feature extraction modules
from src.analysis.thermodynamic_analysis import extract_thermodynamic_features
from src.analysis.dihedral_analysis import extract_dihedral_features
from src.analysis.mutual_information import calculate_mutual_information, convert_mi_to_evolutionary_features
from src.data.extract_features_simple import save_features_npz

# Import memory monitoring utilities
from src.analysis.memory_monitor import MemoryTracker, log_memory_usage, plot_memory_usage
```

### 5.2 Refactored Imports in Notebook

```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

# Import refactored modules
from src.data.data_manager import DataManager
from src.features.feature_extractor import FeatureExtractor
from src.processing.batch_processor import BatchProcessor
from src.processing.memory_monitor import MemoryMonitor
from src.validation.result_validator import ResultValidator
from src.utils.visualization import visualize_rna_structure, plot_mi_matrix
```

### 5.3 Refactored Module Imports

```python
# In data_manager.py
import os
import pandas as pd
import numpy as np
from pathlib import Path

# In feature_extractor.py
import numpy as np
import time
from src.features.thermodynamic_features import extract_thermodynamic_features
from src.features.mutual_information_features import calculate_mutual_information
from src.processing.memory_monitor import MemoryMonitor

# In batch_processor.py
import time
import json
from src.data.data_manager import DataManager
from src.features.feature_extractor import FeatureExtractor
from src.processing.memory_monitor import MemoryMonitor
from src.validation.result_validator import ResultValidator
```

## 6. File Interaction Diagram

This diagram shows how files are read and written by different components:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│   Raw Data      │────────►│   DataManager   │────────►│  Processed Data │
│  (CSV, FASTA)   │         │                 │         │  (NPZ features) │
│                 │         └─────────────────┘         │                 │
└─────────────────┘                  │                  └─────────────────┘
                                     │                            ▲
                                     ▼                            │
                            ┌─────────────────┐                   │
                            │                 │                   │
                            │FeatureExtractor │───────────────────┘
                            │                 │
                            └─────────────────┘
```

## 7. Migration Checklist

To implement the refactoring, follow this checklist:

1. [ ] Create directory structure
2. [ ] Implement DataManager
   - [ ] Move data loading functions
   - [ ] Implement file I/O methods
   - [ ] Add error handling
3. [ ] Implement FeatureExtractor
   - [ ] Move feature extraction functions
   - [ ] Add memory monitoring integration
   - [ ] Implement validation methods
4. [ ] Implement BatchProcessor
   - [ ] Move batch processing functions
   - [ ] Add memory-aware scheduling
   - [ ] Implement progress tracking
5. [ ] Implement MemoryMonitor
   - [ ] Move memory tracking functions
   - [ ] Add visualization methods
   - [ ] Implement memory limit enforcement
6. [ ] Implement ResultValidator
   - [ ] Create validation methods
   - [ ] Implement reporting functions
7. [ ] Create utility modules
   - [ ] Configuration management
   - [ ] Visualization tools
8. [ ] Create test suite
   - [ ] Unit tests for each component
   - [ ] Integration tests
   - [ ] Memory tests
9. [ ] Update notebook
   - [ ] Update imports
   - [ ] Use refactored components
   - [ ] Add documentation
10. [ ] Final validation
    - [ ] Compare outputs with original code
    - [ ] Verify memory usage
    - [ ] Check performance