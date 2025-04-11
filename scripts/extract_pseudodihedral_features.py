#!/usr/bin/env python3
"""
Script to extract pseudodihedral angle features from RNA structures.

This script uses the dihedral_analysis module to calculate and extract
pseudodihedral angle features (eta and theta angles in sin/cos representation)
from RNA structures. It can process individual files or batch process multiple
structures.

Usage:
    python extract_pseudodihedral_features.py --target TARGET_ID [--output OUTPUT_DIR] [--batch BATCH_FILE]
    
Example:
    python extract_pseudodihedral_features.py --target 4V8J_CW --output data/processed/dihedral_features
    python extract_pseudodihedral_features.py --batch data/target_lists/training_list.txt --output data/processed/dihedral_features
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import traceback

# Add the parent directory to the path so we can import the module
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

try:
    from src.analysis.dihedral_analysis import calculate_pseudo_dihedrals, extract_dihedral_features
except ImportError as e:
    print(f"Error importing dihedral_analysis module: {e}")
    sys.exit(1)

def load_structure_data(target_id, data_dir="data/raw"):
    """
    Load structure data for a given target.
    
    Args:
        target_id: Target ID
        data_dir: Directory containing structure data
        
    Returns:
        DataFrame with structure coordinates or None if not found
    """
    # Try to find the structure file
    structure_paths = [
        Path(data_dir) / "structures" / f"{target_id}_coords.csv",
        Path(data_dir) / f"{target_id}_coords.csv",
        Path(data_dir) / "coordinates" / f"{target_id}.csv",
        Path(data_dir) / "coordinates" / f"{target_id}_coords.csv"
    ]
    
    for path in structure_paths:
        if path.exists():
            print(f"Loading structure data from {path}")
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"Error loading structure data: {e}")
                return None
    
    # If we can't find specific structure files, try to extract from the main CSV
    labels_file = Path(data_dir) / "train_labels.csv"
    if labels_file.exists():
        print(f"Loading from main labels file {labels_file}")
        try:
            # Read in the whole CSV
            all_data = pd.read_csv(labels_file)
            
            # Filter rows for this target ID
            target_data = all_data[all_data["ID"].str.startswith(f"{target_id}_")]
            
            if len(target_data) > 0:
                print(f"Found {len(target_data)} residues for {target_id}")
                return target_data
            else:
                print(f"No data found for {target_id} in {labels_file}")
                return None
        except Exception as e:
            print(f"Error loading from labels file: {e}")
            return None
    
    print(f"Could not find structure data for {target_id}")
    return None

def extract_features_for_target(target_id, output_dir=None, data_dir="data/raw"):
    """
    Extract pseudodihedral features for a single target.
    
    Args:
        target_id: Target ID
        output_dir: Directory to save features
        data_dir: Directory containing structure data
        
    Returns:
        Dictionary with extracted features or None if failed
    """
    print(f"\nProcessing target: {target_id}")
    start_time = time.time()
    
    try:
        # Load structure data
        coords_df = load_structure_data(target_id, data_dir)
        if coords_df is None:
            print(f"Failed to load structure data for {target_id}")
            return None
        
        # Calculate pseudodihedral angles
        n_residues = len(coords_df)
        print(f"Calculating pseudodihedral angles for {n_residues} residues")
        
        # Create output file path if output_dir is provided
        output_file = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"{target_id}_dihedral_features.npz"
        
        # Extract dihedral features
        dihedral_features = extract_dihedral_features(coords_df, output_file=output_file, include_raw_angles=True)
        
        # Print summary
        print(f"Extracted features for {target_id}:")
        print(f"- Features shape: {dihedral_features['features'].shape}")
        print(f"- Contains raw angles: {('eta' in dihedral_features) and ('theta' in dihedral_features)}")
        print(f"- Processing time: {time.time() - start_time:.2f} seconds")
        
        return dihedral_features
    
    except Exception as e:
        print(f"Error extracting features for {target_id}: {e}")
        traceback.print_exc()
        return None

def batch_process(target_list, output_dir=None, data_dir="data/raw"):
    """
    Process multiple targets in batch mode.
    
    Args:
        target_list: List of target IDs
        output_dir: Directory to save features
        data_dir: Directory containing structure data
        
    Returns:
        Dictionary with results for each target
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    start_time = time.time()
    print(f"Starting batch processing for {len(target_list)} targets...")
    
    results = {}
    successes = 0
    failures = 0
    
    # Create a summary file
    summary_file = None
    if output_dir:
        summary_file = output_dir / 'batch_summary.json'
    
    try:
        for i, target_id in enumerate(target_list):
            print(f"\nProcessing target {i+1}/{len(target_list)}: {target_id}")
            target_start = time.time()
            
            # Extract features
            result = extract_features_for_target(target_id, output_dir, data_dir)
            
            # Store result if successful
            if result is not None:
                results[target_id] = {
                    'success': True,
                    'n_residues': result['features'].shape[0],
                    'processing_time': time.time() - target_start
                }
                successes += 1
            else:
                results[target_id] = {
                    'success': False,
                    'processing_time': time.time() - target_start
                }
                failures += 1
            
            print(f"Completed {target_id} in {time.time() - target_start:.2f} seconds")
        
        # Generate and save summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_targets': len(target_list),
            'successful': successes,
            'failed': failures,
            'elapsed_time': time.time() - start_time,
            'targets': results
        }
        
        if summary_file:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved batch summary to: {summary_file}")
        
        print(f"\nBatch processing complete:")
        print(f"- Total: {len(target_list)} targets")
        print(f"- Successful: {successes} targets")
        print(f"- Failed: {failures} targets")
        print(f"- Elapsed time: {time.time() - start_time:.2f} seconds")
        
        return results
    
    except Exception as e:
        print(f"Error in batch processing: {e}")
        traceback.print_exc()
        return results

def get_target_ids_from_csv(data_dir="data/raw"):
    """
    Extract list of unique target IDs from the train_labels.csv file.
    
    Args:
        data_dir: Directory containing the CSV file
        
    Returns:
        List of unique target IDs
    """
    labels_file = Path(data_dir) / "train_labels.csv"
    if not labels_file.exists():
        print(f"Error: {labels_file} does not exist")
        return []
    
    try:
        df = pd.read_csv(labels_file)
        
        # Extract target IDs from the ID column (format: TARGET_ID_RESIDUE_NUM)
        target_ids = []
        for id_str in df['ID']:
            # Split the ID string and get the target ID part
            parts = id_str.split('_')
            if len(parts) >= 2:
                target_id = f"{parts[0]}_{parts[1]}"  # Take the first two parts (e.g., "1SCL_A")
                target_ids.append(target_id)
        
        # Get unique target IDs
        unique_targets = sorted(list(set(target_ids)))
        print(f"Found {len(unique_targets)} unique target IDs in {labels_file}")
        return unique_targets
    
    except Exception as e:
        print(f"Error extracting target IDs from {labels_file}: {e}")
        return []

def update_metadata_in_existing_files(directory):
    """
    Update existing NPZ files with feature names and metadata.
    
    Args:
        directory: Directory containing NPZ files to update
    """
    try:
        # Create metadata dictionary
        feature_names = ['eta_sin', 'eta_cos', 'theta_sin', 'theta_cos']
        metadata = {
            'feature_names': feature_names,
            'feature_description': 'Pseudo-dihedral angle features in sin/cos encoding',
            'column_0': 'eta_sin - sine of eta angle',
            'column_1': 'eta_cos - cosine of eta angle',
            'column_2': 'theta_sin - sine of theta angle',
            'column_3': 'theta_cos - cosine of theta angle',
            'metadata_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Find all NPZ files in the directory
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return
            
        npz_files = list(directory.glob("*_dihedral_features.npz"))
        if not npz_files:
            print(f"No NPZ files found in {directory}")
            return
            
        print(f"Found {len(npz_files)} NPZ files to update")
        
        # Process each file
        for i, npz_file in enumerate(npz_files):
            try:
                # Load the existing file
                data = dict(np.load(npz_file, allow_pickle=True))
                
                # Skip if already has metadata
                if 'feature_names' in data:
                    print(f"Skipping {npz_file.name} - already has metadata")
                    continue
                
                # Add metadata
                data['feature_names'] = feature_names
                data['metadata'] = str(metadata)
                
                # Save the updated file
                np.savez_compressed(npz_file, **data)
                print(f"Updated {i+1}/{len(npz_files)}: {npz_file.name}")
                
            except Exception as e:
                print(f"Error updating {npz_file.name}: {e}")
                
        print(f"Metadata update complete: {len(npz_files)} files processed")
        
    except Exception as e:
        print(f"Error in update_metadata_in_existing_files: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Extract pseudodihedral angle features from RNA structures")
    group = parser.add_mutually_exclusive_group(required=False)  # Changed to optional
    group.add_argument("--target", help="Target RNA ID")
    group.add_argument("--batch", help="Path to file with list of targets (one per line)")
    group.add_argument("--all", action="store_true", help="Process all targets in train_labels.csv")
    group.add_argument("--update-metadata", action="store_true", help="Update metadata in existing NPZ files")
    parser.add_argument("--output", help="Output directory for feature files", default="data/processed/dihedral_features")
    parser.add_argument("--data-dir", help="Directory containing structure data", default="data/raw")
    parser.add_argument("--limit", type=int, help="Limit the number of targets to process (for testing)", default=0)
    args = parser.parse_args()
    
    # Handle the update-metadata option
    if args.update_metadata:
        print(f"Updating metadata in existing NPZ files in {args.output}")
        update_metadata_in_existing_files(args.output)
        return
    
    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        print(f"Output will be saved to: {args.output}")
    
    if args.target:
        # Process a single target
        extract_features_for_target(args.target, args.output, args.data_dir)
    elif args.batch:
        # Process targets in batch mode from a file
        try:
            with open(args.batch, 'r') as f:
                targets = [line.strip() for line in f if line.strip()]
            
            print(f"Found {len(targets)} targets in batch file")
            
            # Apply limit if specified
            if args.limit > 0 and args.limit < len(targets):
                print(f"Limiting to first {args.limit} targets")
                targets = targets[:args.limit]
                
            batch_process(targets, args.output, args.data_dir)
        except Exception as e:
            print(f"Error processing batch file: {e}")
            sys.exit(1)
    elif args.all:
        # Process all targets from the CSV file
        targets = get_target_ids_from_csv(args.data_dir)
        
        if not targets:
            print("No targets found in CSV file")
            sys.exit(1)
            
        # Apply limit if specified
        if args.limit > 0 and args.limit < len(targets):
            print(f"Limiting to first {args.limit} targets")
            targets = targets[:args.limit]
            
        print(f"Processing all {len(targets)} targets from CSV file")
        batch_process(targets, args.output, args.data_dir)
    else:
        # No target or batch specified, show help
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()