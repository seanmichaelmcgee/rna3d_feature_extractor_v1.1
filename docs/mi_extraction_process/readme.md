# Mutual Information Extraction Process

This document provides a detailed overview of how mutual information (MI) matrices are being developed in the RNA 3D Explorer Core project, including command-line tools used, modules being called, and configuration files referenced.

## Current MI Extraction Process

The current mutual information extraction process running in the background was initiated with the following command:

```bash
nohup ./scripts/run_mi_small_batches.sh 20 > nohup_msa_extraction.out 2>&1 &
```

This command runs the MI extraction in the background, outputs logs to `nohup_msa_extraction.out`, and processes 20 sequences per batch.

## Command-Line Interface

The core command being executed for each RNA sequence is:

```bash
python src/analysis/rna_mi_pipeline/rna_mi_pipeline.py \
    --input "$MSA_FILE" \
    --output "$OUTPUT_DIR" \
    --max_length $MAX_LENGTH \
    --chunk_size $CHUNK_SIZE \
    --overlap $OVERLAP \
    --max_sequences $MAX_SEQUENCES \
    --workers $WORKERS \
    --parallel \
    --verbose
```

Where the variables are set in the batch script as:
- `MAX_LENGTH`: 500 (Maximum sequence length to process without chunking)
- `CHUNK_SIZE`: 400 (Size of each chunk for long sequences)
- `OVERLAP`: 150 (Overlap between chunks)
- `MAX_SEQUENCES`: 3000 (Maximum number of sequences from MSA to use)
- `WORKERS`: 2 (Number of parallel workers)

## Modules and Implementation

### Key Modules

1. `src/analysis/mutual_information.py`
   - Core implementation of mutual information calculation
   - Contains functions for basic MI matrix computation
   - Implements Average Product Correction (APC) for noise reduction

2. `src/analysis/rna_mi_pipeline/rna_mi_pipeline.py`
   - Main executable script for the pipeline
   - Handles command-line arguments
   - Orchestrates the entire MI extraction process
   - Implements progress tracking and error handling

3. `src/analysis/rna_mi_pipeline/enhanced_mi.py`
   - Advanced MI calculation with RNA-specific optimizations
   - Implements sequence weighting to reduce redundancy bias
   - Provides position filtering based on conservation
   - Contains RNA-specific APC correction algorithm

4. `src/analysis/rna_mi_pipeline/mi_config.py`
   - Configuration parameters and optimization profiles
   - Default settings for different RNA lengths and hardware capabilities
   - Parameter sets for various resource constraints

### Implementation Details

The mutual information calculation follows these steps:

1. **MSA Loading and Filtering**:
   - Loads the Multiple Sequence Alignment (MSA) file
   - Filters sequences based on similarity and gap content
   - Removes positions with excessive gaps

2. **Chunking for Long RNAs**:
   - For sequences longer than MAX_LENGTH, splits into overlapping chunks
   - Each chunk is processed independently
   - Results are later recombined with position-based weighting

3. **MI Calculation**:
   - Computes mutual information between all position pairs
   - Calculates nucleotide frequencies and joint probabilities
   - Applies sequence weighting to reduce bias from redundant sequences

4. **Post-processing**:
   - Applies RNA-specific APC correction
   - Performs mild Gaussian smoothing to reduce noise
   - For chunked processing, recombines results with sophisticated weighting
   - Extracts top-scoring position pairs

5. **Output Generation**:
   - Saves the MI matrix as a numpy array (.npy)
   - Generates visualization plots (heatmap, contact map)
   - Creates a report with statistics and top coupling pairs

## Configuration Profiles

The pipeline uses different configuration profiles optimized for various scenarios:

1. **small**: For short RNAs or limited resources
   - MAX_LENGTH: 300
   - CHUNK_SIZE: 250
   - OVERLAP: 100
   - MAX_SEQUENCES: 2000
   - WORKERS: 1

2. **medium**: Balanced settings (currently used)
   - MAX_LENGTH: 500
   - CHUNK_SIZE: 400
   - OVERLAP: 150
   - MAX_SEQUENCES: 3000
   - WORKERS: 2

3. **large**: For long RNAs or large MSAs
   - MAX_LENGTH: 750
   - CHUNK_SIZE: 600
   - OVERLAP: 200
   - MAX_SEQUENCES: 5000
   - WORKERS: 4

4. **memory**: Memory-optimized for limited resources
   - MAX_LENGTH: 400
   - CHUNK_SIZE: 300
   - OVERLAP: 100
   - MAX_SEQUENCES: 2000
   - WORKERS: 1
   - BATCH_SIZE: 1000

5. **highperf**: High-performance for 32-core systems
   - MAX_LENGTH: 1000
   - CHUNK_SIZE: 800
   - OVERLAP: 300
   - MAX_SEQUENCES: 10000
   - WORKERS: 16
   - BATCH_SIZE: 10000

## Batch Processing System

The batch processing system works as follows:

1. `run_mi_small_batches.sh` script:
   - Takes a batch size parameter (number of sequences per batch)
   - Divides the work into small manageable batches
   - Creates logs for each batch in logs/mi_extraction/
   - Handles sequence list management

2. Batch Execution:
   - Each batch runs in sequence (not in parallel)
   - Progress is tracked in the log file
   - Failed sequences are recorded for later retry

3. Output Structure:
   - MI matrices are saved to data/processed/mi_matrices/
   - Each RNA has its own directory with:
     - mi_matrix.npy: Full mutual information matrix
     - mi_features.csv: Extracted evolutionary coupling features
     - mi_plot.png: Visualization of the MI matrix
     - mi_stats.json: Statistics about the calculation

## Performance Considerations

- **Memory Usage**: The current configuration is optimized for systems with 16GB RAM
- **Processing Time**: Approximately 5-10 minutes per RNA sequence (varies with length)
- **Chunking Strategy**: Crucial for RNAs longer than 500 nucleotides
- **Parallelization**: Limited to 2 workers to avoid memory issues

## Integration with Analysis Tools

The MI matrices are used in several ways within the project:

1. As input features for machine learning models in src/models/
2. For visualization of evolutionary couplings in src/visualization/
3. For validation of predicted 3D structures in src/analysis/structure_analysis.py

## Monitoring and Troubleshooting

The MI extraction process can be monitored through:

1. Main log file: nohup_msa_extraction.out
2. Batch logs: logs/mi_extraction/batch_*.log
3. Progress tracking: data/processed/mi_matrices/progress.json

Common issues and solutions are documented in scripts/README_MI_PIPELINE.md.