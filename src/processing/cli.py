"""
Command-line interface for batch processing RNA targets.

This module provides a command-line interface for running batch processing jobs
using the BatchProcessor class.
"""

import os
import argparse
import logging
import sys
import json
from typing import List, Optional

from src.processing.batch_processor import BatchProcessor
from src.data.data_manager import DataManager
from src.features.feature_extractor import FeatureExtractor
from src.analysis.memory_monitor import MemoryMonitor

def setup_logging(log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
    )

def load_targets_from_file(file_path: str) -> List[str]:
    """Load target IDs from a file, one per line."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_targets_from_csv(file_path: str, id_column: str = "ID") -> List[str]:
    """Load target IDs from a CSV file."""
    import pandas as pd
    df = pd.read_csv(file_path)
    
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in CSV file: {file_path}")
    
    return df[id_column].unique().tolist()

def main():
    """Run the batch processing CLI."""
    parser = argparse.ArgumentParser(description="Process RNA targets for feature extraction.")
    
    # Input options
    input_group = parser.add_argument_group("Input options")
    target_group = input_group.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target", "-t", 
        action="append", 
        help="Target ID to process (can be specified multiple times)"
    )
    target_group.add_argument(
        "--targets-file", "-tf", 
        help="File containing target IDs, one per line"
    )
    target_group.add_argument(
        "--targets-csv", "-tc", 
        help="CSV file containing target IDs in a column"
    )
    input_group.add_argument(
        "--id-column", "-ic", 
        help="Column name for target IDs in CSV file (default: 'ID')",
        default="ID"
    )
    
    # Feature extraction options
    feature_group = parser.add_argument_group("Feature extraction options")
    feature_group.add_argument(
        "--extract-thermo", "-et", 
        action="store_true", 
        help="Extract thermodynamic features"
    )
    feature_group.add_argument(
        "--extract-mi", "-em", 
        action="store_true", 
        help="Extract mutual information features"
    )
    feature_group.add_argument(
        "--pseudocount", "-pc", 
        type=float, 
        help="Pseudocount value for MI calculation",
        default=None
    )
    
    # Batch processing options
    batch_group = parser.add_argument_group("Batch processing options")
    batch_group.add_argument(
        "--batch-name", "-bn", 
        help="Name for this batch processing run",
        default=None
    )
    batch_group.add_argument(
        "--batch-size", "-bs", 
        type=int, 
        help="Number of targets to process in a single batch (default: 10)",
        default=10
    )
    batch_group.add_argument(
        "--max-memory", "-mm", 
        type=float, 
        help="Maximum allowed memory usage in GB (default: 16.0)",
        default=16.0
    )
    batch_group.add_argument(
        "--no-save-intermediates", "-nsi", 
        action="store_true", 
        help="Do not save intermediate batch results"
    )
    batch_group.add_argument(
        "--resume", "-r", 
        help="Resume a previously interrupted batch processing job",
        action="store_true"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output-dir", "-od", 
        help="Directory to save processed features (default: 'data/processed')",
        default="data/processed"
    )
    output_group.add_argument(
        "--log-dir", "-ld", 
        help="Directory to save processing logs (default: 'data/processed/logs')",
        default="data/processed/logs"
    )
    output_group.add_argument(
        "--log-file", "-lf", 
        help="Path to log file",
        default=None
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Get target IDs
    target_ids = []
    if args.target:
        target_ids = args.target
    elif args.targets_file:
        target_ids = load_targets_from_file(args.targets_file)
    elif args.targets_csv:
        target_ids = load_targets_from_csv(args.targets_csv, args.id_column)
    
    logger.info(f"Processing {len(target_ids)} targets")
    
    # Ensure at least one feature type is selected
    if not args.extract_thermo and not args.extract_mi:
        logger.error("At least one feature type must be selected (--extract-thermo or --extract-mi)")
        sys.exit(1)
    
    # Create processor components
    data_manager = DataManager()
    feature_extractor = FeatureExtractor()
    memory_monitor = MemoryMonitor()
    
    # Create batch processor
    batch_processor = BatchProcessor(
        data_manager=data_manager,
        feature_extractor=feature_extractor,
        memory_monitor=memory_monitor,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        max_memory_usage_gb=args.max_memory,
        batch_size=args.batch_size,
    )
    
    # Run batch processing
    try:
        if args.resume and args.batch_name:
            logger.info(f"Resuming batch processing: {args.batch_name}")
            results = batch_processor.resume_batch_processing(
                batch_name=args.batch_name,
                extract_thermo=args.extract_thermo,
                extract_mi=args.extract_mi,
                save_intermediates=not args.no_save_intermediates,
            )
        else:
            logger.info("Starting new batch processing job")
            results = batch_processor.process_targets(
                target_ids=target_ids,
                extract_thermo=args.extract_thermo,
                extract_mi=args.extract_mi,
                save_intermediates=not args.no_save_intermediates,
                batch_name=args.batch_name,
            )
        
        # Print summary
        logger.info(f"Batch processing completed: {results['batch_name']}")
        logger.info(f"Total targets: {results['total_targets']}")
        logger.info(f"Successful targets: {results['successful_targets']}")
        logger.info(f"Skipped targets: {results['skipped_targets']}")
        
    except Exception as e:
        logger.error(f"Error running batch processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()