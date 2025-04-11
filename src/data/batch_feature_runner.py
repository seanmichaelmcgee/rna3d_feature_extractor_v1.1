#!/usr/bin/env python3
"""
High-Performance Batch Feature Runner

This script efficiently processes large batches of RNA sequences for thermodynamic
feature extraction using parallel processing and memory optimization techniques.
It reads a CSV file of RNA sequences, filters by constraints, and extracts features
for each sequence, saving results as .npz files.

Usage example:
  python batch_feature_runner.py \
    --csv data/raw/train_sequences.csv \
    --output-dir data/processed/batch_npz \
    --max-row 715 \
    --length-min 50 \
    --length-max 100 \
    --batch-size 10 \
    --jobs 4

Features:
- Multi-core parallel processing with joblib
- Memory usage monitoring and optimization
- Progress tracking with tqdm
- Error handling with automatic retry
- Flexible output formats (batch or individual files)

"""

import argparse
import sys
import os
import time
import json
import traceback
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd  # Make sure pandas is installed
import numpy as np

# For parallel processing
import multiprocessing
from joblib import Parallel, delayed

# For progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

# For memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Install with 'pip install psutil' for memory monitoring.")

# We assume extract_features_simple.py is in the same directory (src/data)
# Adjust the import if your environment is different
import extract_features_simple as efs

# Memory monitoring class adapted from data-loader-module.py
class MemoryMonitor:
    """Memory usage monitoring utilities."""
    
    @staticmethod
    def get_memory_usage_gb():
        """Get current memory usage in GB."""
        if not HAS_PSUTIL:
            return None
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1e9  # Convert to GB
    
    @staticmethod
    def get_memory_percent():
        """Get memory usage as percentage of system total."""
        if not HAS_PSUTIL:
            return None
            
        process = psutil.Process(os.getpid())
        return process.memory_percent()
    
    @staticmethod
    def print_memory_usage(label="Current"):
        """Print current memory usage."""
        if not HAS_PSUTIL:
            print(f"{label} memory usage: Unknown (psutil not available)")
            return
            
        mem_gb = MemoryMonitor.get_memory_usage_gb()
        mem_percent = MemoryMonitor.get_memory_percent()
        total_mem = psutil.virtual_memory().total / 1e9
        
        print(f"{label} memory usage: {mem_gb:.2f} GB ({mem_percent:.1f}% of process, {mem_gb/total_mem*100:.1f}% of system RAM)")

# Setup logging
def setup_logging(output_dir, verbose=False):
    """
    Configure logging to both console and file.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save log file
    verbose : bool
        Whether to set DEBUG level (otherwise INFO)
    
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("batch_runner")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(console_fmt)
    logger.addHandler(console)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f"batch_processing_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Log file created at: {log_file}")
    return logger

# Timer utility
class Timer:
    """Simple timer for performance tracking."""
    def __init__(self, name="Operation", logger=None):
        self.name = name
        self.logger = logger
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        message = f"{self.name} completed in {self.duration:.4f} seconds"
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

def process_single_sequence(seq_id, sequence, max_retries=2, verbose=False, pf_scale=1.5, 
                         validate_thermo=True, entropy_threshold=1e-6, logger=None):
    """
    Process a single RNA sequence with retry logic and enhanced thermodynamic validation.
    
    Parameters:
    -----------
    seq_id : str
        Unique identifier for the sequence
    sequence : str
        RNA sequence to process
    max_retries : int
        Maximum number of retry attempts on failure
    verbose : bool
        Whether to print detailed progress
    pf_scale : float
        Partition function scaling factor for ViennaRNA calculations
    validate_thermo : bool
        Whether to validate thermodynamic consistency of features
    entropy_threshold : float
        Threshold for detecting zero entropy values
    logger : logging.Logger
        Logger instance for output
        
    Returns:
    --------
    dict
        Dictionary containing extracted features or None if all attempts failed
    """
    features = None
    attempt = 0
    
    log = logger or logging.getLogger("batch_runner")
    
    while attempt <= max_retries and features is None:
        if attempt > 0:
            log.info(f"Retry {attempt}/{max_retries} for {seq_id}")
            # Increase pf_scale on retry for longer sequences (helps with numeric overflow)
            if len(sequence) > 200 and pf_scale < 5.0:
                pf_scale *= 1.5
                log.info(f"Increasing pf_scale to {pf_scale} for retry")
        
        try:
            # Extract features using the core module
            start_time = time.time()
            features = efs.extract_features(sequence, pf_scale=pf_scale)
            extraction_time = time.time() - start_time
            
            # Add metadata
            if features:
                features['seq_id'] = seq_id
                features['processing_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                features['extraction_time'] = extraction_time
                features['pf_scale_used'] = pf_scale
                
                # Check for thermodynamic consistency if requested
                if validate_thermo and 'mfe' in features and 'ensemble_energy' in features:
                    mfe = features.get('mfe', 0.0)
                    ensemble_energy = features.get('ensemble_energy', 0.0)
                    valid = ensemble_energy >= mfe
                    features['thermodynamically_valid'] = valid
                    
                    if not valid:
                        log.warning(f"Thermodynamic inconsistency detected for {seq_id}: " 
                                   f"ensemble energy ({ensemble_energy}) < MFE ({mfe})")
                
                # Log base pair probability matrix statistics if available
                if 'base_pair_probs' in features and hasattr(features['base_pair_probs'], 'size') and features['base_pair_probs'].size > 0:
                    bpp_matrix = features['base_pair_probs']
                    features['bpp_stats'] = {
                        'nonzero_count': int(np.count_nonzero(bpp_matrix)),
                        'max_value': float(np.max(bpp_matrix)),
                        'mean_value': float(np.mean(bpp_matrix))
                    }
                    
                    if verbose:
                        log.debug(f"BPP matrix for {seq_id}: {features['bpp_stats']['nonzero_count']} non-zero entries, "
                                 f"max: {features['bpp_stats']['max_value']:.4f}, mean: {features['bpp_stats']['mean_value']:.4f}")
                
                # Check entropy calculation results
                if 'positional_entropy' in features and hasattr(features['positional_entropy'], 'size') and features['positional_entropy'].size > 0:
                    entropy = features['positional_entropy']
                    features['entropy_stats'] = {
                        'mean': float(np.mean(entropy)),
                        'max': float(np.max(entropy)),
                        'min': float(np.min(entropy)),
                        'zeros': int(np.sum(entropy < entropy_threshold))
                    }
                    
                    # Validate entropy calculation
                    if features['entropy_stats']['max'] < entropy_threshold:
                        log.warning(f"Zero entropy detected for {seq_id}, might indicate calculation issue")
                        
                    # Flag sequences with high percentage of zero entropy positions
                    if len(entropy) > 0 and features['entropy_stats']['zeros'] / len(entropy) > 0.9:
                        log.warning(f"More than 90% zero entropy positions for {seq_id} ({features['entropy_stats']['zeros']}/{len(entropy)})")
                
                log.info(f"Successfully processed {seq_id} (length: {len(sequence)}) in {extraction_time:.2f} seconds")
                if verbose:
                    log.debug(f"Features for {seq_id}: {list(features.keys())}")
        except Exception as e:
            error_msg = f"Error processing {seq_id}: {str(e)}"
            log.error(error_msg)
            if verbose:
                log.debug(traceback.format_exc())
            
            # Clear memory before retry
            if HAS_PSUTIL:
                gc_before = MemoryMonitor.get_memory_usage_gb()
                import gc
                gc.collect()
                gc_after = MemoryMonitor.get_memory_usage_gb()
                if gc_before and gc_after:
                    log.debug(f"GC freed {gc_before - gc_after:.2f} GB")
        
        attempt += 1
    
    if features is None:
        log.warning(f"Failed to process {seq_id} after {max_retries+1} attempts")
    
    return features

def dynamic_batch_sizing(df, initial_batch_size, min_batch_size=5, max_memory_percent=70):
    """
    Dynamically determine optimal batch size based on available memory.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the sequences to process
    initial_batch_size : int
        Starting batch size to consider
    min_batch_size : int
        Minimum batch size regardless of memory
    max_memory_percent : int
        Maximum percentage of system RAM to use
        
    Returns:
    --------
    int
        Optimal batch size based on available memory
    """
    if not HAS_PSUTIL:
        return initial_batch_size
    
    # Get current memory usage
    current_memory_percent = psutil.virtual_memory().percent
    available_memory_percent = 100 - current_memory_percent
    
    # If we have plenty of memory, use the initial batch size
    if available_memory_percent > max_memory_percent:
        return initial_batch_size
    
    # Calculate a reduced batch size proportional to available memory
    reduced_batch_size = int(initial_batch_size * (available_memory_percent / max_memory_percent))
    
    # Ensure we don't go below the minimum
    return max(min_batch_size, reduced_batch_size)

def save_batch_stats(output_dir, batch_stats, logger=None):
    """Save batch processing statistics to a JSON file."""
    log = logger or logging.getLogger("batch_runner")
    stats_file = output_dir / "batch_processing_stats.json"
    
    # Convert any non-serializable objects to strings
    for batch in batch_stats:
        for key, value in batch.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                batch[key] = str(value)
    
    with open(stats_file, 'w') as f:
        json.dump(batch_stats, f, indent=2)
    
    log.info(f"Saved processing statistics to {stats_file}")

def save_checkpoint(output_dir, processed_count, batch_stats, all_stats, logger=None):
    """
    Save checkpoint data to resume processing later.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save checkpoint file
    processed_count : int
        Number of sequences processed so far
    batch_stats : list
        List of batch processing statistics
    all_stats : dict
        Overall processing statistics
    logger : logging.Logger
        Logger instance
    """
    log = logger or logging.getLogger("batch_runner")
    checkpoint_file = output_dir / "checkpoint.json"
    
    checkpoint_data = {
        "processed_count": processed_count,
        "batch_stats": batch_stats,
        "all_stats": all_stats,
        "checkpoint_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Convert any non-serializable objects to strings
    for section in ["batch_stats", "all_stats"]:
        data = checkpoint_data[section]
        if isinstance(data, list):
            for item in data:
                for key, value in item.items():
                    if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        item[key] = str(value)
        elif isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    data[key] = str(value)
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    log.info(f"Saved checkpoint at {processed_count} sequences")

def load_checkpoint(output_dir, logger=None):
    """
    Load checkpoint data to resume processing.
    
    Parameters:
    -----------
    output_dir : Path
        Directory containing checkpoint file
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    tuple or None
        (processed_count, batch_stats, all_stats) if checkpoint exists, None otherwise
    """
    log = logger or logging.getLogger("batch_runner")
    checkpoint_file = output_dir / "checkpoint.json"
    
    if not checkpoint_file.exists():
        log.info("No checkpoint file found, starting from beginning")
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        processed_count = checkpoint_data.get("processed_count", 0)
        batch_stats = checkpoint_data.get("batch_stats", [])
        all_stats = checkpoint_data.get("all_stats", {})
        checkpoint_time = checkpoint_data.get("checkpoint_time", "unknown")
        
        log.info(f"Loaded checkpoint from {checkpoint_time} with {processed_count} sequences processed")
        return processed_count, batch_stats, all_stats
    except Exception as e:
        log.error(f"Error loading checkpoint: {e}")
        return None

def main():
    # Start with basic logging to console, will be enhanced after parsing args
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("batch_runner")
    
    parser = argparse.ArgumentParser(
        description="High-performance batch processor for RNA feature extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the CSV file with RNA sequences")
    parser.add_argument("--output-dir", type=str, default="data/processed/batch_npz",
                        help="Directory to save NPZ output files")
    parser.add_argument("--max-row", type=int, default=715,
                        help="Do not process rows beyond this index (1-based)")
    parser.add_argument("--start-row", type=int, default=0,
                        help="Start processing from this row index (0-based)")
    parser.add_argument("--length-min", type=int, default=50,
                        help="Minimum sequence length")
    parser.add_argument("--length-max", type=int, default=100,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of sequences to process per batch")
    parser.add_argument("--id-col", type=str, default="id",
                        help="CSV column containing unique sequence identifiers")
    parser.add_argument("--seq-col", type=str, default="sequence",
                        help="CSV column containing the RNA sequence")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of parallel jobs (default: CPU count - 1)")
    parser.add_argument("--output-format", type=str, choices=["batch", "individual", "both"], 
                        default="batch", help="How to save output files")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Dynamically adjust batch size based on memory usage")
    parser.add_argument("--max-memory-percent", type=int, default=70,
                        help="Maximum percentage of system RAM to use")
    parser.add_argument("--retry", type=int, default=2,
                        help="Maximum retry attempts for failed sequences")
    parser.add_argument("--pf-scale", type=float, default=1.5,
                        help="Partition function scaling factor for ViennaRNA (higher values like 1.5-3.0 for long sequences)")
    parser.add_argument("--validate-thermo", action="store_true", default=True,
                        help="Validate thermodynamic consistency of features")
    parser.add_argument("--no-validate-thermo", action="store_false", dest="validate_thermo",
                        help="Skip thermodynamic validation (faster, but may miss errors)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress messages")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Number of batches to process before saving checkpoint data")
    parser.add_argument("--resume", action="store_true",
                        help="Resume processing from the last checkpoint if available")
    parser.add_argument("--entropy-threshold", type=float, default=1e-6,
                        help="Threshold for detecting zero entropy values (default: 1e-6)")
    args = parser.parse_args()

    # Determine number of parallel jobs
    if args.jobs is None:
        args.jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Convert output dir to Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup proper logging
    logger = setup_logging(output_dir, args.verbose)

    # Individual output directory if needed
    if args.output_format in ["individual", "both"]:
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)
    
    # Setup summary statistics tracking - will be overwritten if resuming
    batch_stats = []
    all_stats = {
        "total_sequences": 0,
        "successful_sequences": 0,
        "failed_sequences": 0,
        "total_processing_time": 0,
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Starting point for processing
    processed_count = 0
    
    # Check for resume option
    if args.resume:
        checkpoint_data = load_checkpoint(output_dir, logger)
        if checkpoint_data:
            processed_count, batch_stats, all_stats = checkpoint_data
            
            # Update start time to include resumed session
            all_stats["resume_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # If we're using start-row, adjust processed_count
            if args.start_row > 0 and processed_count == 0:
                logger.info(f"Starting from row {args.start_row} based on command-line parameter")
            elif processed_count > 0:
                logger.info(f"Resuming from checkpoint at sequence {processed_count}")
                if args.start_row > 0:
                    logger.warning(f"Ignoring --start-row {args.start_row} in favor of checkpoint at {processed_count}")

    # Print configuration
    logger.info("\n=== RNA Feature Extraction Batch Processing ===")
    logger.info(f"CSV File: {args.csv}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Row Range: {args.start_row} to {args.max_row}")
    logger.info(f"Sequence Length Range: {args.length_min} to {args.length_max} nucleotides")
    logger.info(f"Batch Size: {args.batch_size}" + (" (dynamic)" if args.dynamic_batch else ""))
    logger.info(f"Parallel Jobs: {args.jobs}")
    logger.info(f"Output Format: {args.output_format}")
    logger.info(f"PF Scale: {args.pf_scale}")
    logger.info(f"Thermodynamic Validation: {'Enabled' if args.validate_thermo else 'Disabled'}")
    logger.info(f"Entropy Threshold: {args.entropy_threshold}")
    logger.info(f"Checkpoint Interval: {args.checkpoint_interval} batches")
    logger.info(f"Max Retries: {args.retry}")
    
    # Print initial memory usage
    if HAS_PSUTIL:
        mem_gb = MemoryMonitor.get_memory_usage_gb()
        mem_percent = MemoryMonitor.get_memory_percent()
        total_mem = psutil.virtual_memory().total / 1e9
        
        logger.info(f"Initial memory usage: {mem_gb:.2f} GB ({mem_percent:.1f}% of process, {mem_gb/total_mem*100:.1f}% of system RAM)")

    # Read and filter CSV
    with Timer("CSV reading and filtering", logger):
        try:
            df = pd.read_csv(args.csv)
            print(f"Read {len(df)} sequences from CSV")
            
            # Handle start-row and max-row parameters
            total_rows = len(df)
            
            # Apply start-row filter (0-based indexing)
            if args.start_row > 0:
                if args.start_row >= total_rows:
                    print(f"Error: start-row {args.start_row} is beyond the total number of rows ({total_rows})")
                    sys.exit(1)
                df = df.iloc[args.start_row:]
                print(f"Starting from row {args.start_row} (0-based indexing)")
            
            # Filter out rows beyond max_row (assuming 1-based indexing in user's statement)
            if args.max_row < total_rows:
                # Calculate how many rows to keep after start_row
                rows_to_keep = args.max_row - args.start_row
                if rows_to_keep <= 0:
                    print(f"Error: max-row {args.max_row} is less than or equal to start-row {args.start_row}")
                    sys.exit(1)
                    
                if rows_to_keep < len(df):
                    df = df.iloc[:rows_to_keep]
                    print(f"Processing {rows_to_keep} rows (from {args.start_row} to {args.max_row})")

            # Filter by sequence length
            def length_in_range(seq):
                if not isinstance(seq, str):
                    return False
                seq_len = len(str(seq).replace('T', 'U').upper())
                return args.length_min <= seq_len <= args.length_max

            pre_length_count = len(df)
            df = df[df[args.seq_col].apply(length_in_range)]
            print(f"Filtered by length: {pre_length_count} -> {len(df)} sequences " + 
                  f"({len(df)/pre_length_count*100:.1f}% remaining)")
            
            # Check if we have valid IDs
            if args.id_col not in df.columns:
                print(f"Warning: ID column '{args.id_col}' not found in CSV.")
                print(f"Available columns: {', '.join(df.columns)}")
                print("Using row indices as IDs")
                df['_generated_id'] = [f"seq_{i}" for i in range(len(df))]
                args.id_col = '_generated_id'
                
            # Check if we have valid sequences
            if args.seq_col not in df.columns:
                print(f"Error: Sequence column '{args.seq_col}' not found in CSV.")
                print(f"Available columns: {', '.join(df.columns)}")
                sys.exit(1)
                
            # Ensure all IDs are unique
            if df[args.id_col].duplicated().any():
                print(f"Warning: Duplicate IDs found in {args.id_col} column")
                print("Adding unique suffix to duplicate IDs")
                # Add a suffix to make IDs unique
                df = df.assign(**{
                    args.id_col: df.groupby(args.id_col).cumcount().astype(str) + '_' + df[args.id_col]
                })
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            traceback.print_exc()
            sys.exit(1)

    # At this point, df contains only rows up to max_row
    # whose sequences are within the specified length range
    total_sequences = len(df)
    all_stats["total_sequences"] = total_sequences
    
    if total_sequences == 0:
        logger.error("No sequences to process after filtering. Exiting.")
        sys.exit(0)
    
    # Calculate initial batch count
    batch_size = args.batch_size
    batch_count = (total_sequences + batch_size - 1) // batch_size

    logger.info(f"\nProcessing {total_sequences} sequences in approximately {batch_count} batches")
    logger.info(f"Using {args.jobs} parallel workers\n")

    # If we're resuming, skip already processed sequences
    if processed_count > 0:
        logger.info(f"Skipping {processed_count} already processed sequences")
        if processed_count >= total_sequences:
            logger.info("All sequences were already processed. Nothing to do.")
            sys.exit(0)
    
    # Prepare progress bar if available
    if HAS_TQDM:
        remaining = total_sequences - processed_count
        pbar = tqdm(total=remaining, desc="Processing sequences", initial=processed_count)
    
    # Process each batch
    overall_start_time = time.time()
    
    # Get initial batch size (may be adjusted dynamically)
    current_batch_size = batch_size
    
    while processed_count < total_sequences:
        # Dynamically adjust batch size if requested
        if args.dynamic_batch and HAS_PSUTIL:
            new_batch_size = dynamic_batch_sizing(
                df, batch_size, 
                min_batch_size=max(1, batch_size // 4),
                max_memory_percent=args.max_memory_percent
            )
            
            if new_batch_size != current_batch_size:
                logger.info(f"Adjusting batch size: {current_batch_size} → {new_batch_size} " +
                      f"(memory usage: {psutil.virtual_memory().percent}%)")
                current_batch_size = new_batch_size
        
        # Calculate batch range
        start_row = processed_count
        end_row = min(start_row + current_batch_size, total_sequences)
        batch_df = df.iloc[start_row:end_row]
        
        batch_idx = processed_count // batch_size + 1
        batch_size_actual = len(batch_df)
        
        logger.info(f"\n=== Processing batch {batch_idx}/{batch_count} " +
                   f"(rows {start_row+1} to {end_row}) ===")
        logger.info(f"Batch size: {batch_size_actual} sequences")

        # Start timing the batch
        batch_start_time = time.time()
        
        # Process sequences in parallel
        try:
            # Prepare input for parallel processing
            sequence_inputs = [
                (str(row[args.id_col]), str(row[args.seq_col]))
                for _, row in batch_df.iterrows()
            ]
            
            # Process in parallel with progress tracking
            if HAS_TQDM:
                # For tqdm, we process each sequence individually
                batch_results = []
                for seq_id, sequence in sequence_inputs:
                    result = process_single_sequence(
                        seq_id, sequence, 
                        max_retries=args.retry, 
                        verbose=args.verbose,
                        pf_scale=args.pf_scale,
                        validate_thermo=args.validate_thermo,
                        entropy_threshold=args.entropy_threshold,
                        logger=logger
                    )
                    batch_results.append(result)
                    pbar.update(1)
            else:
                # Without tqdm, use joblib's parallel processing
                batch_results = Parallel(n_jobs=args.jobs)(
                    delayed(process_single_sequence)(
                        seq_id, sequence, 
                        max_retries=args.retry, 
                        verbose=args.verbose,
                        pf_scale=args.pf_scale,
                        validate_thermo=args.validate_thermo,
                        entropy_threshold=args.entropy_threshold,
                        logger=logger
                    )
                    for seq_id, sequence in sequence_inputs
                )
        
            # Count successful extractions
            successful = sum(1 for r in batch_results if r is not None)
            failed = batch_size_actual - successful
            
            # Update statistics
            all_stats["successful_sequences"] += successful
            all_stats["failed_sequences"] += failed
            
            # Track entropy and thermodynamic validation statistics
            batch_thermo_invalid = 0
            batch_entropy_zero = 0
            batch_entropy_mostly_zero = 0
            
            for r in batch_results:
                if r is not None:
                    # Check for thermodynamic validation
                    if 'thermodynamically_valid' in r and r['thermodynamically_valid'] is False:
                        batch_thermo_invalid += 1
                        
                    # Check for entropy issues
                    if 'entropy_stats' in r:
                        if r['entropy_stats'].get('max', 0) < args.entropy_threshold:
                            batch_entropy_zero += 1
                        if (r['entropy_stats'].get('zeros', 0) / len(r.get('positional_entropy', [])) > 0.9) if len(r.get('positional_entropy', [])) > 0 else False:
                            batch_entropy_mostly_zero += 1
            
            # Add to batch statistics
            all_stats["thermo_invalid_count"] = all_stats.get("thermo_invalid_count", 0) + batch_thermo_invalid
            all_stats["entropy_zero_count"] = all_stats.get("entropy_zero_count", 0) + batch_entropy_zero
            all_stats["entropy_mostly_zero_count"] = all_stats.get("entropy_mostly_zero_count", 0) + batch_entropy_mostly_zero
            
            # Calculate batch processing time
            batch_time = time.time() - batch_start_time
            avg_time_per_seq = batch_time / batch_size_actual if batch_size_actual > 0 else 0
            
            # Record batch statistics
            batch_stat = {
                "batch_number": batch_idx,
                "processed": batch_size_actual,
                "successful": successful,
                "failed": failed,
                "processing_time": batch_time,
                "avg_time_per_sequence": avg_time_per_seq,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "validation_stats": {
                    "thermo_invalid": batch_thermo_invalid,
                    "entropy_zero": batch_entropy_zero,
                    "entropy_mostly_zero": batch_entropy_mostly_zero
                }
            }
            
            if HAS_PSUTIL:
                batch_stat["memory_usage_gb"] = MemoryMonitor.get_memory_usage_gb()
                batch_stat["memory_percent"] = MemoryMonitor.get_memory_percent()
            
            batch_stats.append(batch_stat)
            
            # Print batch summary
            logger.info(f"\nBatch {batch_idx} summary:")
            logger.info(f"- Processed: {batch_size_actual} sequences")
            logger.info(f"- Successful: {successful} ({successful/batch_size_actual*100:.1f}%)")
            logger.info(f"- Failed: {failed}")
            logger.info(f"- Time: {batch_time:.2f} seconds ({avg_time_per_seq:.2f} sec/sequence)")
            
            # Add validation statistics to summary if any issues were found
            if batch_thermo_invalid > 0:
                logger.info(f"- Thermodynamic inconsistencies: {batch_thermo_invalid}")
            if batch_entropy_zero > 0:
                logger.info(f"- Zero entropy sequences: {batch_entropy_zero}")
            if batch_entropy_mostly_zero > 0:
                logger.info(f"- Sequences with >90% zero entropy: {batch_entropy_mostly_zero}")
            
            if HAS_PSUTIL:
                mem_gb = MemoryMonitor.get_memory_usage_gb()
                mem_percent = MemoryMonitor.get_memory_percent()
                total_mem = psutil.virtual_memory().total / 1e9
                
                logger.info(f"- Memory usage: {mem_gb:.2f} GB ({mem_percent:.1f}% of system RAM)")
            
            # Save results
            # Option 1: Save as a single NPZ for the batch
            if args.output_format in ["batch", "both"]:
                save_dict = {}
                for idx, features in enumerate(batch_results):
                    if features is None:
                        continue
                        
                    # Get sequence ID
                    seq_id = features.get('seq_id', f"seq_{idx}")
                    
                    # Store each feature with a prefix
                    for key, value in features.items():
                        save_dict[f"{seq_id}_{key}"] = value
                
                # Build the output file name
                batch_output_file = output_dir / f"batch_{batch_idx}_of_{batch_count}.npz"
                
                # Save as NPZ
                try:
                    np.savez_compressed(batch_output_file, **save_dict)
                    logger.info(f"Saved batch {batch_idx} NPZ to {batch_output_file}")
                except Exception as e:
                    logger.error(f"Error saving batch NPZ: {e}")
                    if args.verbose:
                        logger.debug(traceback.format_exc())
            
            # Option 2: Save individual NPZ files for each sequence
            if args.output_format in ["individual", "both"]:
                for features in batch_results:
                    if features is None:
                        continue
                        
                    # Get sequence ID
                    seq_id = features.get('seq_id', "unknown")
                    
                    # Build the output file name
                    seq_output_file = individual_dir / f"{seq_id}_features.npz"
                    
                    # Save as NPZ
                    try:
                        np.savez_compressed(seq_output_file, **features)
                        if args.verbose:
                            logger.debug(f"Saved individual NPZ for {seq_id}")
                    except Exception as e:
                        logger.error(f"Error saving individual NPZ for {seq_id}: {e}")
                
                logger.info(f"Saved {successful} individual NPZ files to {individual_dir}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            logger.debug(traceback.format_exc())
        
        # Update processed count
        processed_count += batch_size_actual
        
        # Save batch statistics and checkpoint data periodically
        if batch_idx % args.checkpoint_interval == 0 or processed_count >= total_sequences:
            try:
                # Save batch stats
                save_batch_stats(output_dir, batch_stats, logger)
                
                # Save checkpoint for resuming
                save_checkpoint(output_dir, processed_count, batch_stats, all_stats, logger)
            except Exception as e:
                logger.error(f"Warning: Could not save checkpoint data: {e}")
    
    # Close progress bar if used
    if HAS_TQDM:
        pbar.close()
    
    # Calculate overall processing time
    overall_time = time.time() - overall_start_time
    all_stats["total_processing_time"] = overall_time
    all_stats["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_stats["avg_time_per_sequence"] = overall_time / total_sequences if total_sequences > 0 else 0
    
    # Print final summary
    logger.info("\n=== Batch processing complete ===")
    logger.info(f"Total sequences processed: {total_sequences}")
    logger.info(f"Successfully extracted features: {all_stats['successful_sequences']} ({all_stats['successful_sequences']/total_sequences*100:.1f}%)")
    logger.info(f"Failed sequences: {all_stats['failed_sequences']}")
    logger.info(f"Total processing time: {overall_time:.2f} seconds")
    logger.info(f"Average time per sequence: {all_stats['avg_time_per_sequence']:.4f} seconds")
    
    # Add quality control summary
    logger.info("\nQuality control summary:")
    thermo_invalid = all_stats.get("thermo_invalid_count", 0) 
    entropy_zero = all_stats.get("entropy_zero_count", 0)
    entropy_mostly_zero = all_stats.get("entropy_mostly_zero_count", 0)
    
    if all_stats['successful_sequences'] > 0:
        logger.info(f"- Thermodynamic inconsistencies: {thermo_invalid} ({thermo_invalid/all_stats['successful_sequences']*100:.1f}%)")
        logger.info(f"- Zero entropy sequences: {entropy_zero} ({entropy_zero/all_stats['successful_sequences']*100:.1f}%)")
        logger.info(f"- Sequences with >90% zero entropy: {entropy_mostly_zero} ({entropy_mostly_zero/all_stats['successful_sequences']*100:.1f}%)")
    
    # Add warning if significant quality issues
    if all_stats['successful_sequences'] > 0:
        if thermo_invalid / all_stats['successful_sequences'] > 0.1:
            logger.warning("⚠️ High rate of thermodynamic inconsistencies detected")
        if entropy_zero / all_stats['successful_sequences'] > 0.1:
            logger.warning("⚠️ High rate of zero entropy sequences detected")
    
    if HAS_PSUTIL:
        mem_gb = MemoryMonitor.get_memory_usage_gb()
        mem_percent = MemoryMonitor.get_memory_percent()
        total_mem = psutil.virtual_memory().total / 1e9
        
        logger.info(f"Final memory usage: {mem_gb:.2f} GB ({mem_percent:.1f}% of system RAM)")
    
    # Save overall statistics
    summary_file = output_dir / "processing_summary.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"Saved processing summary to {summary_file}")
    except Exception as e:
        logger.error(f"Warning: Could not save processing summary: {e}")
        
    # Cleanup checkpoint if processing completed successfully
    if processed_count >= total_sequences and all_stats["failed_sequences"] == 0:
        checkpoint_file = output_dir / "checkpoint.json"
        if checkpoint_file.exists():
            try:
                os.rename(checkpoint_file, output_dir / "completed_checkpoint.json")
                logger.info("Processing completed successfully, renamed checkpoint file")
            except Exception:
                pass

if __name__ == "__main__":
    main()