#!/usr/bin/env python3
"""
RNA Evolutionary Coupling Pipeline

This script implements the complete pipeline for RNA evolutionary coupling
analysis using Mutual Information with APC correction and chunking for
long sequences.

Usage:
    python rna_mi_pipeline.py --input input_dir --output output_dir [options]

Author: Your Name
Date: April 2025
"""

import os
import sys
import argparse
import logging
import glob
import time
import multiprocessing
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import from enhanced_mi module
from enhanced_mi import (
    process_rna_msa_for_structure,
    load_msa_robust,
    filter_rna_msa,
    chunk_and_analyze_rna,
    calculate_mutual_information_enhanced,
    apply_rna_apc_correction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rna_mi_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('rna_mi_pipeline')

def process_single_rna(msa_file, output_dir, params):
    """
    Process a single RNA MSA file.
    
    Parameters:
    -----------
    msa_file : str
        Path to MSA file
    output_dir : str
        Output directory for features
    params : dict
        Processing parameters
        
    Returns:
    --------
    dict
        Processing results
    """
    try:
        # Create output paths
        rna_id = os.path.basename(msa_file).split('.')[0]
        features_file = os.path.join(output_dir, f"{rna_id}_features.npz")
        
        # Log start of processing
        logger.info(f"Processing RNA {rna_id} from {msa_file}")
        start_time = time.time()
        
        # Process RNA
        features = process_rna_msa_for_structure(
            msa_file=msa_file,
            output_features=features_file,
            max_length=params['max_length'],
            chunk_size=params['chunk_size'],
            overlap=params['overlap'],
            gap_threshold=params['gap_threshold'],
            identity_threshold=params['identity_threshold'],
            max_sequences=params['max_sequences'],
            conservation_range=params['conservation_range'],
            parallel=params['parallel'],
            n_jobs=params['n_jobs'],
            verbose=params['verbose']
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if features:
            logger.info(f"Successfully processed {rna_id} in {processing_time:.2f} seconds")
            return {
                'rna_id': rna_id,
                'status': 'success',
                'features_file': features_file,
                'processing_time': processing_time,
                'method': features['method'],
                'sequence_length': features['sequence_length'],
                'sequence_count': features['sequence_count']
            }
        else:
            logger.error(f"Failed to process {rna_id}")
            return {
                'rna_id': rna_id,
                'status': 'failed',
                'error': 'Processing returned None'
            }
            
    except Exception as e:
        logger.error(f"Error processing {msa_file}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'rna_id': os.path.basename(msa_file).split('.')[0],
            'status': 'error',
            'error': str(e)
        }

def process_rna_dataset(input_dir, output_dir, params, max_workers=None):
    """
    Process all RNA MSA files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing MSA files
    output_dir : str
        Output directory for features
    params : dict
        Processing parameters
    max_workers : int, optional
        Maximum number of parallel workers
        
    Returns:
    --------
    dict
        Processing results for all RNAs
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all MSA files
    msa_files = []
    for ext in ['.fasta', '.fa', '.afa', '.msa']:
        msa_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not msa_files:
        logger.error(f"No MSA files found in {input_dir}")
        return {'status': 'failed', 'error': 'No MSA files found'}
    
    logger.info(f"Found {len(msa_files)} MSA files to process")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(msa_files))
    
    # Use parallel processing if multiple workers
    if max_workers > 1 and len(msa_files) > 1:
        logger.info(f"Processing {len(msa_files)} RNAs using {max_workers} workers")
        
        with multiprocessing.Pool(max_workers) as pool:
            results = pool.starmap(
                process_single_rna,
                [(msa_file, output_dir, params) for msa_file in msa_files]
            )
    else:
        # Process sequentially
        logger.info(f"Processing {len(msa_files)} RNAs sequentially")
        results = [process_single_rna(msa_file, output_dir, params) for msa_file in msa_files]
    
    # Combine results
    all_results = {r['rna_id']: r for r in results}
    
    # Calculate summary statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    if success_count > 0:
        avg_time = sum(r.get('processing_time', 0) for r in results if r['status'] == 'success') / success_count
        avg_length = sum(r.get('sequence_length', 0) for r in results if r['status'] == 'success') / success_count
    else:
        avg_time = 0
        avg_length = 0
    
    # Create summary report
    summary = {
        'status': 'success',
        'total_rnas': len(results),
        'successful': success_count,
        'failed': error_count,
        'success_rate': success_count / len(results) if results else 0,
        'avg_processing_time': avg_time,
        'avg_sequence_length': avg_length,
        'results': all_results
    }
    
    # Save summary report
    summary_file = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"RNA Evolutionary Coupling Processing Summary\n")
        f.write(f"=========================================\n\n")
        f.write(f"Total RNAs: {len(results)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {error_count}\n")
        f.write(f"Success rate: {summary['success_rate']*100:.1f}%\n\n")
        
        f.write(f"Average processing time: {avg_time:.2f} seconds\n")
        f.write(f"Average sequence length: {avg_length:.1f} nucleotides\n\n")
        
        f.write(f"Processing parameters:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write(f"Individual RNA Results:\n")
        for rna_id, result in all_results.items():
            status = result['status']
            if status == 'success':
                f.write(f"  {rna_id}: {status} ({result['sequence_length']} nt, {result['processing_time']:.2f}s)\n")
            else:
                f.write(f"  {rna_id}: {status} - {result.get('error', 'Unknown error')}\n")
    
    logger.info(f"Saved processing summary to {summary_file}")
    logger.info(f"Successfully processed {success_count}/{len(results)} RNAs ({summary['success_rate']*100:.1f}%)")
    
    return summary

def create_performance_visualization(summary, output_file):
    """
    Create visualization of processing performance.
    
    Parameters:
    -----------
    summary : dict
        Processing summary
    output_file : str
        Output visualization file
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        # Extract data
        results = [r for r in summary['results'].values() if r['status'] == 'success']
        if not results:
            logger.warning("No successful results for visualization")
            return False
        
        # Extract sequence lengths and processing times
        seq_lengths = [r['sequence_length'] for r in results]
        proc_times = [r['processing_time'] for r in results]
        
        # Create figure with multiple panels
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Sequence length distribution
        axes[0, 0].hist(seq_lengths, bins=20, alpha=0.7)
        axes[0, 0].set_xlabel('Sequence Length (nt)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('RNA Sequence Length Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Processing time distribution
        axes[0, 1].hist(proc_times, bins=20, alpha=0.7)
        axes[0, 1].set_xlabel('Processing Time (s)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Processing Time Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sequence length vs processing time
        axes[1, 0].scatter(seq_lengths, proc_times, alpha=0.7)
        axes[1, 0].set_xlabel('Sequence Length (nt)')
        axes[1, 0].set_ylabel('Processing Time (s)')
        axes[1, 0].set_title('Processing Time vs Sequence Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Try to fit a polynomial curve
        try:
            from scipy.optimize import curve_fit
            
            def poly_func(x, a, b, c):
                return a * x**2 + b * x + c
            
            popt, _ = curve_fit(poly_func, seq_lengths, proc_times)
            x_range = np.linspace(min(seq_lengths), max(seq_lengths), 100)
            axes[1, 0].plot(x_range, poly_func(x_range, *popt), 'r-', 
                          label=f'Fit: {popt[0]:.2e}x² + {popt[1]:.2f}x + {popt[2]:.2f}')
            axes[1, 0].legend()
        except:
            logger.warning("Failed to fit polynomial curve")
        
        # Plot 4: Success rate by sequence length
        # Group by bins of 100nt
        bin_size = 100
        max_len = max(seq_lengths)
        bins = list(range(0, max_len + bin_size, bin_size))
        success_rates = []
        
        for i in range(len(bins)-1):
            min_len = bins[i]
            max_len = bins[i+1]
            
            # Count all RNAs in this length range
            all_in_range = [r for r in summary['results'].values() 
                          if r.get('sequence_length', 0) >= min_len and 
                             r.get('sequence_length', 0) < max_len]
            
            # Count successful RNAs in this range
            success_in_range = [r for r in all_in_range if r['status'] == 'success']
            
            # Calculate success rate
            if all_in_range:
                rate = len(success_in_range) / len(all_in_range)
                success_rates.append((min_len, max_len, rate))
        
        # Plot success rates
        if success_rates:
            x = [(min_len + max_len) / 2 for min_len, max_len, _ in success_rates]
            y = [rate for _, _, rate in success_rates]
            axes[1, 1].bar(x, y, width=bin_size*0.8, alpha=0.7)
            axes[1, 1].set_xlabel('Sequence Length (nt)')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_title('Success Rate by Sequence Length')
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f"RNA MI Pipeline Performance\n{summary['successful']}/{summary['total_rnas']} RNAs Processed Successfully", 
                    fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance visualization to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RNA Evolutionary Coupling Pipeline")
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Input MSA file or directory')
    parser.add_argument('--output', required=True, help='Output directory for features')
    
    # Processing parameters
    parser.add_argument('--max_length', type=int, default=750,
                       help='Maximum sequence length to process without chunking')
    parser.add_argument('--chunk_size', type=int, default=600,
                       help='Size of each chunk for long sequences')
    parser.add_argument('--overlap', type=int, default=200,
                       help='Overlap between chunks')
    parser.add_argument('--gap_threshold', type=float, default=0.5,
                       help='Maximum gap frequency for filtering')
    parser.add_argument('--identity_threshold', type=float, default=0.8,
                       help='Sequence identity threshold for filtering')
    parser.add_argument('--max_sequences', type=int, default=5000,
                       help='Maximum number of sequences to use from MSA')
    parser.add_argument('--conservation_min', type=float, default=0.2,
                       help='Minimum conservation for position filtering')
    parser.add_argument('--conservation_max', type=float, default=0.95,
                       help='Maximum conservation for position filtering')
    
    # Execution parameters
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')
    parser.add_argument('--parallel', action='store_true', 
                       help='Enable parallel processing for MI calculation')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Disable performance visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create parameter dictionary
    params = {
        'max_length': args.max_length,
        'chunk_size': args.chunk_size,
        'overlap': args.overlap,
        'gap_threshold': args.gap_threshold,
        'identity_threshold': args.identity_threshold,
        'max_sequences': args.max_sequences,
        'conservation_range': (args.conservation_min, args.conservation_max),
        'parallel': args.parallel,
        'n_jobs': None,  # Use default
        'verbose': args.verbose
    }
    
    # Log parameters
    logger.info("RNA MI Pipeline starting with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Check if input is a single file or directory
    if os.path.isfile(args.input):
        # Process single RNA
        logger.info(f"Processing single RNA from {args.input}")
        
        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)
        
        # Process RNA
        result = process_single_rna(args.input, args.output, params)
        
        if result['status'] == 'success':
            logger.info(f"Successfully processed RNA to {result['features_file']}")
            sys.exit(0)
        else:
            logger.error(f"Failed to process RNA: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        # Process all RNAs in directory
        logger.info(f"Processing all RNAs in {args.input}")
        
        # Process dataset
        summary = process_rna_dataset(args.input, args.output, params, max_workers=args.workers)
        
        # Create visualization if enabled
        if not args.no_visualization:
            viz_file = os.path.join(args.output, "processing_performance.png")
            create_performance_visualization(summary, viz_file)
        
        # Check overall success
        if summary['status'] == 'success' and summary['success_rate'] > 0.5:
            logger.info("Processing completed successfully")
            sys.exit(0)
        else:
            logger.warning(f"Processing completed with issues: "
                         f"{summary['successful']}/{summary['total_rnas']} RNAs processed successfully")
            sys.exit(1)

if __name__ == "__main__":
    main()