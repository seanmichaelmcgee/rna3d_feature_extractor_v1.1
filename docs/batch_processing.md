# Batch Processing Guide

This guide explains how to use the BatchProcessor class for processing multiple RNA targets.

## Overview

The BatchProcessor class coordinates the processing of multiple RNA targets, managing data loading, feature extraction, memory monitoring, and results saving. It provides a convenient interface for batch processing with features like:

- Memory usage monitoring to prevent out-of-memory errors
- Incremental processing with batch size limits
- Resumable batch processing
- Detailed error handling and logging
- Comprehensive results tracking

## Basic Usage

```python
from src.processing.batch_processor import BatchProcessor
from src.data.data_manager import DataManager
from src.features.feature_extractor import FeatureExtractor
from src.analysis.memory_monitor import MemoryMonitor

# Create processor components
data_manager = DataManager()
feature_extractor = FeatureExtractor()
memory_monitor = MemoryMonitor()

# Create batch processor
batch_processor = BatchProcessor(
    data_manager=data_manager,
    feature_extractor=feature_extractor,
    memory_monitor=memory_monitor,
    output_dir="data/processed",
    log_dir="data/processed/logs",
    max_memory_usage_gb=16.0,
    batch_size=10,
)

# Process targets
target_ids = ["R1001", "R1002", "R1003", "R1004", "R1005"]
results = batch_processor.process_targets(
    target_ids=target_ids,
    extract_thermo=True,
    extract_mi=True,
    save_intermediates=True,
    batch_name="my_batch",
)

# Print summary
print(f"Batch processing completed: {results['batch_name']}")
print(f"Total targets: {results['total_targets']}")
print(f"Successful targets: {results['successful_targets']}")
print(f"Skipped targets: {results['skipped_targets']}")
```

## Command-Line Interface

The batch processor can also be used from the command line:

```bash
# Process a single target
python -m src.processing.cli --target R1001 --extract-thermo --extract-mi

# Process multiple targets
python -m src.processing.cli --target R1001 --target R1002 --extract-thermo --extract-mi

# Process targets from a file
python -m src.processing.cli --targets-file targets.txt --extract-thermo --extract-mi

# Process targets from a CSV file
python -m src.processing.cli --targets-csv data.csv --id-column ID --extract-thermo --extract-mi

# Set batch processing parameters
python -m src.processing.cli --targets-file targets.txt --extract-thermo --extract-mi \
    --batch-name my_batch --batch-size 5 --max-memory 8.0 \
    --output-dir data/processed --log-dir data/processed/logs

# Resume a previously interrupted batch
python -m src.processing.cli --resume --batch-name my_batch --extract-thermo --extract-mi
```

## Resuming Batch Processing

If a batch processing job is interrupted, it can be resumed:

```python
# Resume a previously interrupted batch
batch_processor.resume_batch_processing(
    batch_name="my_batch",
    extract_thermo=True,
    extract_mi=True,
    save_intermediates=True,
)
```

## Memory Management

The batch processor monitors memory usage during processing. If memory usage exceeds the specified limit, the processor will skip the current target and continue with the next one. This helps prevent out-of-memory errors.

You can adjust the memory limit with the `max_memory_usage_gb` parameter:

```python
batch_processor = BatchProcessor(
    # ... other parameters ...
    max_memory_usage_gb=32.0,  # Increase memory limit to 32 GB
)
```

## Batch Size

The batch processor processes targets in smaller batches to manage memory more effectively. You can adjust the batch size with the `batch_size` parameter:

```python
batch_processor = BatchProcessor(
    # ... other parameters ...
    batch_size=20,  # Process 20 targets at a time
)
```

## Output Files

The batch processor generates the following output files:

- Feature files: Saved in the output directory (e.g., `data/processed/thermo_features/R1001_thermo_features.npz`)
- Batch parameters: Saved in the log directory (e.g., `data/processed/logs/my_batch_params.json`)
- Batch results: Saved in the log directory (e.g., `data/processed/logs/my_batch_results.json`)
- Intermediate results: Saved in the log directory (e.g., `data/processed/logs/my_batch_1_results.json`)
- Target IDs: Saved in the log directory (e.g., `data/processed/logs/my_batch_1_targets.txt`)

These files can be used for results analysis and to resume interrupted batch processing jobs.