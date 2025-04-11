#\!/usr/bin/env python3
"""
NPZ to CSV Converter for RNA Features

This script converts NPZ feature files to CSV format for easier inspection
and analysis. It can process individual NPZ files or batch NPZ files with
multiple sequences.

Usage:
  python npz_to_csv.py --input data/processed/batch_npz/batch_1_of_1.npz --output features.csv
  python npz_to_csv.py --input data/processed/batch_npz/individual/ --output all_features.csv
  python npz_to_csv.py --input data/processed/batch_npz/individual/seq_0_features.npz --output seq_0.csv
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def npz_to_dict(npz_file):
    """Convert an NPZ file to a dictionary with proper scalar handling."""
    with np.load(npz_file, allow_pickle=True) as data:
        # Create a dictionary with all the data
        result = {}
        for key in data.files:
            value = data[key]
            
            # Convert numpy scalar types to Python native types
            if isinstance(value, np.ndarray) and value.ndim == 0:
                # Handle scalars (0-d arrays)
                result[key] = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 1 and value.size < 100:
                # Handle small 1D arrays - convert to list
                result[key] = value.tolist()
            elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] < 20 and value.shape[1] < 20:
                # Handle small 2D arrays - flatten to string representation
                # This is just for CSV display purposes
                result[key] = str(value.tolist())
            elif isinstance(value, np.ndarray):
                # For larger arrays, just store dimensions
                result[key] = f"Array({value.shape}, {value.dtype})"
            else:
                # Other types (strings, etc.)
                result[key] = value
        
        return result

def process_individual_npz(file_path):
    """Process a single NPZ file and return a dictionary representation."""
    try:
        # Get ID from filename or contained data
        file_name = Path(file_path).stem
        
        # Load data
        data_dict = npz_to_dict(file_path)
        
        # Add metadata
        data_dict['file_source'] = str(file_path)
        
        # For index rows - ensure ID is determined by seq_id field if present
        if 'seq_id' in data_dict:
            # Use the embedded sequence ID
            return data_dict
        else:
            # Use filename as ID
            data_dict['seq_id'] = file_name
            return data_dict
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_batch_npz(file_path):
    """Process a batch NPZ file containing multiple sequences and return a list of dictionaries."""
    results = []
    
    try:
        # Load the batch NPZ file
        with np.load(file_path, allow_pickle=True) as data:
            # Group keys by sequence ID
            sequences = {}
            
            for key in data.files:
                # Batch NPZ stores keys as "seq_id_featurename"
                parts = key.split('_', 1)
                if len(parts) >= 2:
                    seq_id = parts[0]
                    feature_name = parts[1]
                    
                    if seq_id not in sequences:
                        sequences[seq_id] = {}
                    
                    # Extract the value
                    value = data[key]
                    
                    # Convert numpy scalars to Python types
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        sequences[seq_id][feature_name] = value.item()
                    elif isinstance(value, np.ndarray) and value.ndim == 1 and value.size < 100:
                        sequences[seq_id][feature_name] = value.tolist()
                    elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] < 20 and value.shape[1] < 20:
                        sequences[seq_id][feature_name] = str(value.tolist())
                    elif isinstance(value, np.ndarray):
                        sequences[seq_id][feature_name] = f"Array({value.shape}, {value.dtype})"
                    else:
                        sequences[seq_id][feature_name] = value
            
            # Convert each sequence's data to a dictionary
            for seq_id, seq_data in sequences.items():
                seq_data['seq_id'] = seq_id
                seq_data['file_source'] = str(file_path)
                results.append(seq_data)
                
    except Exception as e:
        print(f"Error processing batch file {file_path}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ feature files to CSV for easier inspection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to NPZ file or directory containing NPZ files')
    parser.add_argument('--output', type=str, default='features.csv',
                        help='Path to output CSV file')
    parser.add_argument('--select-features', type=str,
                        help='Comma-separated list of features to include (default: all)')
    parser.add_argument('--exclude-arrays', action='store_true',
                        help='Exclude large array features like base_pair_probs from output')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Create output directory if needed
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Track all features data
    all_data = []
    
    # Process input paths
    if input_path.is_dir():
        # Process all NPZ files in directory
        npz_files = list(input_path.glob('**/*.npz'))
        print(f"Found {len(npz_files)} NPZ files in {input_path}")
        
        for file_path in npz_files:
            if args.verbose:
                print(f"Processing {file_path}")
            
            if 'batch_' in file_path.name:
                # This is a batch file with multiple sequences
                batch_data = process_batch_npz(file_path)
                if batch_data:
                    all_data.extend(batch_data)
                    if args.verbose:
                        print(f"  Added {len(batch_data)} sequences from batch file")
            else:
                # This is an individual sequence file
                seq_data = process_individual_npz(file_path)
                if seq_data:
                    all_data.append(seq_data)
    else:
        # Process single file
        if not input_path.exists():
            print(f"Error: Input file {input_path} not found")
            sys.exit(1)
        
        if 'batch_' in input_path.name:
            # This is a batch file with multiple sequences
            batch_data = process_batch_npz(input_path)
            if batch_data:
                all_data.extend(batch_data)
                print(f"Processed batch file with {len(batch_data)} sequences")
        else:
            # This is an individual sequence file
            seq_data = process_individual_npz(input_path)
            if seq_data:
                all_data.append(seq_data)
                print(f"Processed individual file: {input_path}")
    
    # If no data was loaded, exit
    if not all_data:
        print("No data was loaded. Check your input files.")
        sys.exit(1)
    
    # Filter features if requested
    if args.select_features:
        selected_features = args.select_features.split(',')
        for i, data in enumerate(all_data):
            all_data[i] = {k: v for k, v in data.items() if k in selected_features or k == 'seq_id'}
    
    # Exclude array data if requested
    if args.exclude_arrays:
        for i, data in enumerate(all_data):
            all_data[i] = {k: v for k, v in data.items() 
                          if not (isinstance(v, str) and v.startswith('Array(')) and 
                             not k.endswith('_probs') and
                             not k.endswith('_matrix')}
    
    # Convert to dataframe
    df = pd.DataFrame(all_data)
    
    # Ensure seq_id is the first column
    if 'seq_id' in df.columns:
        cols = ['seq_id'] + [col for col in df.columns if col != 'seq_id']
        df = df[cols]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} sequences with {len(df.columns)} features to {output_path}")
    
    # Print feature summary
    print("\nFeature columns:")
    for col in df.columns[:10]:
        print(f"- {col}")
    
    if len(df.columns) > 10:
        print(f"... and {len(df.columns) - 10} more columns")

if __name__ == "__main__":
    main()
