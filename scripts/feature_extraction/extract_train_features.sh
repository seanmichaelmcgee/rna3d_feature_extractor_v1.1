#!/bin/bash
# Extract all features for train data
# This script mirrors the functionality of the train_features_extraction.ipynb notebook
#
# Usage: ./extract_train_features.sh [OPTIONS]
#
# Options:
#   --csv FILE          CSV file with sequences (default: data/raw/train_sequences.csv)
#   --output-dir DIR    Output directory (default: data/processed)
#   --limit N           Limit processing to N targets (for testing)
#   --cores N           Number of cores to use (default: all - 1)
#   --pf-scale SCALE    Partition function scaling factor (default: 1.5)
#   --target ID         Process only the specified target
#   --targets FILE      Process targets listed in file (one per line)
#   --skip-existing     Skip targets with existing feature files
#   --resume            Resume from last successful target
#   --batch-size N      Number of targets to process before cleanup (default: 5)
#   --memory-limit GB   Maximum memory usage in GB (default: 80% of system memory)
#   --verbose           Enable verbose output
#   --report            Generate HTML report after processing
#   --force             Overwrite existing output files
#   --help              Show this help message

# Set default parameters
CSV_FILE="data/raw/train_sequences.csv"
OUTPUT_DIR="data/processed"
LIMIT=0  # 0 means no limit
CORES=$(( $(nproc) - 1 ))
PF_SCALE=1.5
TARGET=""
TARGETS_FILE=""
SKIP_EXISTING=false
RESUME=false
BATCH_SIZE=5
MEMORY_LIMIT=$(awk '/MemTotal/ {printf "%.0f", $2 * 0.8 / 1024 / 1024}' /proc/meminfo)
VERBOSE=false
REPORT=false
FORCE=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --csv)
      CSV_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --cores)
      CORES="$2"
      shift 2
      ;;
    --pf-scale)
      PF_SCALE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --targets)
      TARGETS_FILE="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=true
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --memory-limit)
      MEMORY_LIMIT="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --report)
      REPORT=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --help)
      echo "Usage: ./extract_train_features.sh [OPTIONS]"
      echo "Extract features for train RNA data"
      echo ""
      echo "Options:"
      echo "  --csv FILE          CSV file with sequences (default: data/raw/train_sequences.csv)"
      echo "  --output-dir DIR    Output directory (default: data/processed)"
      echo "  --limit N           Limit processing to N targets (for testing)"
      echo "  --cores N           Number of cores to use (default: all - 1)"
      echo "  --pf-scale SCALE    Partition function scaling factor (default: 1.5)"
      echo "  --target ID         Process only the specified target"
      echo "  --targets FILE      Process targets listed in file (one per line)"
      echo "  --skip-existing     Skip targets with existing feature files"
      echo "  --resume            Resume from last successful target"
      echo "  --batch-size N      Number of targets to process before cleanup (default: 5)"
      echo "  --memory-limit GB   Maximum memory usage in GB (default: 80% of system memory)"
      echo "  --verbose           Enable verbose output"
      echo "  --report            Generate HTML report after processing"
      echo "  --force             Overwrite existing output files"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Error: Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if output directory exists
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/thermo_features"
mkdir -p "$OUTPUT_DIR/dihedral_features"
mkdir -p "$OUTPUT_DIR/mi_features"
mkdir -p "$OUTPUT_DIR/reports"

# Initialize progress tracking
PROGRESS_FILE="$OUTPUT_DIR/logs/train_progress.json"
if [[ "$RESUME" == "true" && -f "$PROGRESS_FILE" ]]; then
  echo "Resuming from existing progress file"
else
  # Create new progress file
  echo "{\"started\": \"$(date -Iseconds)\", \"processed_targets\": [], \"failed_targets\": [], \"skipped_targets\": []}" > "$PROGRESS_FILE"
fi

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate rna3d-core

echo "=== RNA train Feature Extraction ==="
echo "Starting extraction at $(date)"
echo "  CSV file: $CSV_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Cores: $CORES"
echo "  PF scale: $PF_SCALE"
echo "  Batch size: $BATCH_SIZE"
echo "  Memory limit: ${MEMORY_LIMIT}GB"
[[ "$VERBOSE" == "true" ]] && echo "  Verbose: enabled"
[[ "$RESUME" == "true" ]] && echo "  Resume: enabled"
[[ "$SKIP_EXISTING" == "true" ]] && echo "  Skip existing: enabled"
[[ "$REPORT" == "true" ]] && echo "  Report: enabled"
[[ "$FORCE" == "true" ]] && echo "  Force: enabled"

# Load already processed targets if resuming
PROCESSED_TARGETS=()
FAILED_TARGETS=()
if [[ "$RESUME" == "true" && -f "$PROGRESS_FILE" ]]; then
  readarray -t PROCESSED_TARGETS < <(jq -r '.processed_targets[]' "$PROGRESS_FILE" 2>/dev/null || echo "")
  readarray -t FAILED_TARGETS < <(jq -r '.failed_targets[]' "$PROGRESS_FILE" 2>/dev/null || echo "")
  
  if [[ ${#PROCESSED_TARGETS[@]} -gt 0 ]]; then
    echo "Resuming from previous run. Already processed ${#PROCESSED_TARGETS[@]} targets."
    echo "Previously failed targets: ${#FAILED_TARGETS[@]}"
  fi
fi

# Define Python script to run the feature extraction
cat > "$OUTPUT_DIR/logs/train_extraction_script.py" << 'PYTHONSCRIPT'
#!/usr/bin/env python3
# This script is auto-generated from the train_features_extraction.ipynb notebook

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import psutil
import argparse
import gc
import traceback

# Import feature extraction modules
from src.analysis.thermodynamic_analysis import extract_thermodynamic_features
from src.analysis.dihedral_analysis import extract_dihedral_features
from src.analysis.mutual_information import calculate_mutual_information, convert_mi_to_evolutionary_features
from src.data.extract_features_simple import save_features_npz
from src.analysis.memory_monitor import MemoryTracker, log_memory_usage

# Helper functions from the notebook
def load_rna_data(csv_path):
    """
    Load RNA data from CSV file.
    
    Args:
        csv_path: Path to CSV file containing RNA data
        
    Returns:
        DataFrame with RNA data
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} entries from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def get_unique_target_ids(df, id_col="ID"):
    """
    Extract unique target IDs from dataframe.
    
    Args:
        df: DataFrame with RNA data
        id_col: Column containing IDs
        
    Returns:
        List of unique target IDs
    """
    # Extract target IDs (format: TARGET_ID_RESIDUE_NUM)
    target_ids = []
    for id_str in df[id_col]:
        # Split the ID string and get the target ID part
        parts = id_str.split('_')
        if len(parts) >= 2:
            target_id = f"{parts[0]}_{parts[1]}"  # Take the first two parts (e.g., "1SCL_A")
            target_ids.append(target_id)
    
    # Get unique target IDs
    unique_targets = sorted(list(set(target_ids)))
    print(f"Found {len(unique_targets)} unique target IDs")
    return unique_targets

def load_structure_data(target_id, data_dir=Path("data/raw")):
    """
    Load structure data for a given target from labels CSV.
    
    Args:
        target_id: Target ID
        data_dir: Directory containing data
        
    Returns:
        DataFrame with structure coordinates or None if not found
    """
    data_dir = Path(data_dir)
    
    # Define possible label files (train or validation)
    label_files = [
        data_dir / "train_labels.csv",
        data_dir / "validation_labels.csv"
    ]
    
    for label_file in label_files:
        if label_file.exists():
            try:
                print(f"Looking for {target_id} in {label_file}")
                # Read the entire CSV file
                all_data = pd.read_csv(label_file)
                
                # Filter rows for this target ID
                target_data = all_data[all_data["ID"].str.startswith(f"{target_id}_")]
                
                if len(target_data) > 0:
                    print(f"Found {len(target_data)} residues for {target_id}")
                    return target_data
            except Exception as e:
                print(f"Error loading from {label_file}: {e}")
    
    print(f"Could not find structure data for {target_id} in any labels file")
    return None

def load_msa_data(target_id, data_dir=Path("data/raw")):
    """
    Load MSA data for a given target.
    
    Args:
        target_id: Target ID
        data_dir: Directory containing MSA data
        
    Returns:
        List of MSA sequences or None if not found
    """
    # Try to find the MSA file
    msa_paths = [
        data_dir / "MSA" / f"{target_id}.MSA.fasta",
        data_dir / f"{target_id}.MSA.fasta",
        data_dir / "alignments" / f"{target_id}.MSA.fasta"
    ]
    
    for path in msa_paths:
        if path.exists():
            print(f"Loading MSA data from {path}")
            try:
                # Parse FASTA file
                sequences = []
                current_seq = ""
                 
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append(current_seq)
                                current_seq = ""
                        else:
                            current_seq += line
                            
                    # Add the last sequence
                    if current_seq:
                        sequences.append(current_seq)
                
                print(f"Loaded {len(sequences)} sequences from MSA")
                return sequences
            except Exception as e:
                print(f"Error loading MSA data: {e}")
                return None
    
    print(f"Could not find MSA data for {target_id}")
    return None

def get_sequence_for_target(target_id, data_dir=Path("data/raw")):
    """
    Get RNA sequence for a target ID from the sequence file.
    
    Args:
        target_id: Target ID
        data_dir: Directory containing sequence data
        
    Returns:
        RNA sequence as string or None if not found
    """
    # Try train_sequences.csv first (primary source for training sequences)
    train_seq_path = data_dir / "train_sequences.csv"
    if train_seq_path.exists():
        try:
            df = pd.read_csv(train_seq_path)
            # Try different possible column names for ID and sequence
            id_cols = ["target_id", "ID", "id"]
            seq_cols = ["sequence", "Sequence", "seq"]
            
            for id_col in id_cols:
                if id_col in df.columns:
                    for seq_col in seq_cols:
                        if seq_col in df.columns:
                            # Find the target in the dataframe
                            target_row = df[df[id_col] == target_id]
                            if len(target_row) > 0:
                                sequence = target_row[seq_col].iloc[0]
                                print(f"Found sequence for {target_id} in train_sequences.csv, length: {len(sequence)}")
                                return sequence
        except Exception as e:
            print(f"Error loading sequence data from {train_seq_path}: {e}")
    
    # If not found in train_sequences.csv, try other sequence files
    sequence_paths = [
        data_dir / "sequences.csv",
        data_dir / "test_sequences.csv",
        data_dir / "validation_sequences.csv"
    ]
    
    for path in sequence_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                
                # Try different possible column names
                id_cols = ["target_id", "ID", "id"]
                seq_cols = ["sequence", "Sequence", "seq"]
                
                for id_col in id_cols:
                    if id_col in df.columns:
                        for seq_col in seq_cols:
                            if seq_col in df.columns:
                                # Find the target in the dataframe
                                target_row = df[df[id_col] == target_id]
                                if len(target_row) > 0:
                                    sequence = target_row[seq_col].iloc[0]
                                    print(f"Found sequence for {target_id} in {path.name}, length: {len(sequence)}")
                                    return sequence
            except Exception as e:
                print(f"Error loading sequence data from {path}: {e}")
    
    # As a last resort, try to extract it from MSA data
    print(f"Could not find sequence in CSV files, trying MSA files as a last resort")
    msa_sequences = load_msa_data(target_id, data_dir)
    if msa_sequences and len(msa_sequences) > 0:
        # The first sequence in the MSA is typically the target sequence
        sequence = msa_sequences[0]
        print(f"Found sequence for {target_id} in MSA file, length: {len(sequence)}")
        return sequence
    
    print(f"Could not find sequence for {target_id} in any file")
    return None

def extract_thermo_features_for_target(target_id, sequence=None):
    """
    Extract thermodynamic features for a given target.
    
    Args:
        target_id: Target ID
        sequence: RNA sequence (optional, will be loaded if not provided)
        
    Returns:
        Dictionary with thermodynamic features or None if failed
    """
    print(f"Extracting thermodynamic features for {target_id}")
    start_time = time.time()
    
    try:
        # Get sequence if not provided
        if sequence is None:
            sequence = get_sequence_for_target(target_id)
            if sequence is None:
                print(f"Failed to get sequence for {target_id}")
                return None
        
        # Log initial memory usage
        log_memory_usage(f"Before thermo features for {target_id} (len={len(sequence)})")
        
        # Calculate features with memory monitoring
        print(f"Calculating thermodynamic features for sequence of length {len(sequence)}")
        with MemoryTracker(f"Thermodynamic features calculation for {target_id}"):
            features = extract_thermodynamic_features(sequence)
        
        # Save features
        thermo_dir = Path(args.output_dir) / "thermo_features"
        output_file = thermo_dir / f"{target_id}_thermo_features.npz"
        features['target_id'] = target_id
        features['sequence'] = sequence
        
        with MemoryTracker("Saving thermodynamic features"):
            save_features_npz(features, output_file)
        
        # Log final memory usage
        log_memory_usage(f"After thermo features for {target_id}")
        
        elapsed_time = time.time() - start_time
        print(f"Extracted thermodynamic features in {elapsed_time:.2f} seconds")
        return features
    
    except Exception as e:
        print(f"Error extracting thermodynamic features for {target_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_dihedral_features_for_target(target_id, structure_data=None):
    """
    Extract pseudodihedral angle features for a given target.
    
    Args:
        target_id: Target ID
        structure_data: DataFrame with structure coordinates (optional, will be loaded if not provided)
        
    Returns:
        Dictionary with dihedral features or None if failed
    """
    print(f"Extracting dihedral features for {target_id}")
    start_time = time.time()
    
    try:
        # Get structure data if not provided
        if structure_data is None:
            structure_data = load_structure_data(target_id)
            if structure_data is None:
                print(f"Failed to get structure data for {target_id}")
                return None
        
        # Check if we have at least 4 residues (required for dihedral angles)
        if len(structure_data) < 4:
            print(f"Not enough residues ({len(structure_data)}) for {target_id}, minimum 4 required for dihedral angles")
            return None
        
        # Check if we have the necessary coordinate columns
        required_cols = ['x_1', 'y_1', 'z_1']
        if not all(col in structure_data.columns for col in required_cols):
            # Try to find alternative column names
            alt_x_cols = [col for col in structure_data.columns if col.startswith('x_')]
            if alt_x_cols:
                x_col = alt_x_cols[0]
                y_col = x_col.replace('x_', 'y_')
                z_col = x_col.replace('x_', 'z_')
                
                # Rename columns for compatibility
                if all(col in structure_data.columns for col in [x_col, y_col, z_col]):
                    structure_data = structure_data.rename(columns={
                        x_col: 'x_1',
                        y_col: 'y_1',
                        z_col: 'z_1'
                    })
                    print(f"Renamed columns {x_col}, {y_col}, {z_col} to x_1, y_1, z_1")
                else:
                    print(f"Missing coordinate columns for {target_id}")
                    return None
            else:
                print(f"Missing coordinate columns for {target_id}")
                return None
        
        # Log memory before dihedral calculation
        log_memory_usage(f"Before dihedral features for {target_id} (residues={len(structure_data)})")
        
        # Calculate dihedral features
        dihedral_dir = Path(args.output_dir) / "dihedral_features"
        output_file = dihedral_dir / f"{target_id}_dihedral_features.npz"
        print(f"Calculating dihedral features for {len(structure_data)} residues")
        
        with MemoryTracker(f"Dihedral features calculation for {target_id}"):
            dihedral_features = extract_dihedral_features(structure_data, output_file=output_file, include_raw_angles=True)
        
        # Log memory after dihedral calculation
        log_memory_usage(f"After dihedral features for {target_id}")
        
        elapsed_time = time.time() - start_time
        print(f"Extracted dihedral features in {elapsed_time:.2f} seconds")
        
        # Add target ID
        dihedral_features['target_id'] = target_id
        return dihedral_features
    
    except Exception as e:
        print(f"Error extracting dihedral features for {target_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_mi_features_for_target(target_id, structure_data=None, msa_sequences=None):
    """
    Extract Mutual Information features for a given target.
    
    Args:
        target_id: Target ID
        structure_data: DataFrame with structure data for correlation calculation (optional)
        msa_sequences: List of MSA sequences (optional, will be loaded if not provided)
        
    Returns:
        Dictionary with MI features or None if failed
    """
    print(f"Extracting MI features for {target_id}")
    start_time = time.time()
    
    try:
        # Get MSA sequences if not provided
        if msa_sequences is None:
            msa_sequences = load_msa_data(target_id)
            if msa_sequences is None or len(msa_sequences) < 2:
                print(f"Failed to get MSA data for {target_id} or not enough sequences")
                return None
        
        # Get structure data if not provided (for correlation calculation)
        if structure_data is None and target_id is not None:
            structure_data = load_structure_data(target_id)
        
        # Log memory before MI calculation
        sequence_length = len(msa_sequences[0]) if msa_sequences else 0
        msa_size = len(msa_sequences) if msa_sequences else 0
        log_memory_usage(f"Before MI features for {target_id} (seq_len={sequence_length}, msa_size={msa_size})")
        
        # Calculate MI (this may take some time for large MSAs)
        print(f"Calculating MI for {len(msa_sequences)} sequences")
        with MemoryTracker(f"MI calculation for {target_id}"):
            mi_result = calculate_mutual_information(msa_sequences, verbose=args.verbose)
        
        if mi_result is None:
            print(f"Failed to calculate MI for {target_id}")
            return None
        
        # Convert to evolutionary features
        mi_dir = Path(args.output_dir) / "mi_features"
        output_file = mi_dir / f"{target_id}_mi_features.npz"
        
        # If we have structure data, use it for correlation calculation
        if structure_data is not None:
            print(f"Converting MI to evolutionary features with structural correlation")
            with MemoryTracker(f"MI-structure correlation for {target_id}"):
                features = convert_mi_to_evolutionary_features(mi_result, structure_data, output_file=output_file)
        else:
            print(f"Converting MI to evolutionary features without structural correlation")
            features = mi_result
            
            # Save manually if convert_mi_to_evolutionary_features wasn't used
            if output_file is not None:
                with MemoryTracker(f"Saving MI features for {target_id}"):
                    np.savez_compressed(output_file, **features)
                print(f"Saved MI features to {output_file}")
        
        # Log memory after MI calculation
        log_memory_usage(f"After MI features for {target_id}")
        
        elapsed_time = time.time() - start_time
        print(f"Extracted MI features in {elapsed_time:.2f} seconds")
        
        # Add target ID
        features['target_id'] = target_id
        return features
    
    except Exception as e:
        print(f"Error extracting MI features for {target_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_target(target_id, extract_thermo=True, extract_dihedral=True, extract_mi=True):
    """
    Process a single target, extracting all requested feature types.
    
    Args:
        target_id: Target ID
        extract_thermo: Whether to extract thermodynamic features
        extract_dihedral: Whether to extract dihedral features
        extract_mi: Whether to extract MI features
        
    Returns:
        Dictionary with results for each feature type
    """
    print(f"\nProcessing target: {target_id}")
    results = {'target_id': target_id}
    start_time = time.time()
    
    # Load common data that might be used by multiple feature types
    sequence = get_sequence_for_target(target_id) if extract_thermo else None
    structure_data = load_structure_data(target_id) if extract_dihedral or extract_mi else None
    msa_sequences = load_msa_data(target_id) if extract_mi else None
    
    # Extract thermodynamic features
    if extract_thermo:
        thermo_dir = Path(args.output_dir) / "thermo_features"
        thermo_file = thermo_dir / f"{target_id}_thermo_features.npz"
        
        if thermo_file.exists() and not args.force:
            print(f"Thermodynamic features already exist for {target_id}")
            results['thermo'] = {'success': True, 'skipped': True}
        else:
            thermo_features = extract_thermo_features_for_target(target_id, sequence)
            results['thermo'] = {'success': thermo_features is not None}
    
    # Extract dihedral features
    if extract_dihedral:
        dihedral_dir = Path(args.output_dir) / "dihedral_features"
        dihedral_file = dihedral_dir / f"{target_id}_dihedral_features.npz"
        
        if dihedral_file.exists() and not args.force:
            print(f"Dihedral features already exist for {target_id}")
            results['dihedral'] = {'success': True, 'skipped': True}
        else:
            dihedral_features = extract_dihedral_features_for_target(target_id, structure_data)
            results['dihedral'] = {'success': dihedral_features is not None}
    
    # Extract MI features
    if extract_mi:
        mi_dir = Path(args.output_dir) / "mi_features"
        mi_file = mi_dir / f"{target_id}_mi_features.npz"
        
        if mi_file.exists() and not args.force:
            print(f"MI features already exist for {target_id}")
            results['mi'] = {'success': True, 'skipped': True}
        else:
            mi_features = extract_mi_features_for_target(target_id, structure_data, msa_sequences)
            results['mi'] = {'success': mi_features is not None}
    
    # Calculate total time
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    print(f"Completed processing {target_id} in {elapsed_time:.2f} seconds")
    
    return results

def batch_process_targets(target_ids, extract_thermo=True, extract_dihedral=True, extract_mi=True):
    """
    Process multiple targets in batch mode.
    
    Args:
        target_ids: List of target IDs
        extract_thermo: Whether to extract thermodynamic features
        extract_dihedral: Whether to extract dihedral features
        extract_mi: Whether to extract MI features
        
    Returns:
        Dictionary with results for each target
    """
    print(f"Starting batch processing for {len(target_ids)} targets")
    start_time = time.time()
    
    results = {}
    for i, target_id in enumerate(target_ids):
        print(f"\nProcessing target {i+1}/{len(target_ids)}: {target_id}")
        
        # Process the target
        target_results = process_target(
            target_id, 
            extract_thermo=extract_thermo, 
            extract_dihedral=extract_dihedral, 
            extract_mi=extract_mi
        )
        
        # Store results
        results[target_id] = target_results
    
    # Calculate statistics
    total_time = time.time() - start_time
    
    success_counts = {
        'thermo': sum(1 for r in results.values() if 'thermo' in r and r['thermo']['success']),
        'dihedral': sum(1 for r in results.values() if 'dihedral' in r and r['dihedral']['success']),
        'mi': sum(1 for r in results.values() if 'mi' in r and r['mi']['success'])
    }
    
    skipped_counts = {
        'thermo': sum(1 for r in results.values() if 'thermo' in r and r['thermo'].get('skipped', False)),
        'dihedral': sum(1 for r in results.values() if 'dihedral' in r and r['dihedral'].get('skipped', False)),
        'mi': sum(1 for r in results.values() if 'mi' in r and r['mi'].get('skipped', False))
    }
    
    # Print summary
    print("\nBatch processing complete!")
    print(f"Total targets: {len(target_ids)}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if extract_thermo:
        print(f"Thermodynamic features: {success_counts['thermo']} successful ({skipped_counts['thermo']} skipped)")
        
    if extract_dihedral:
        print(f"Dihedral features: {success_counts['dihedral']} successful ({skipped_counts['dihedral']} skipped)")
        
    if extract_mi:
        print(f"MI features: {success_counts['mi']} successful ({skipped_counts['mi']} skipped)")
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_targets': len(target_ids),
        'total_time': total_time,
        'success_counts': success_counts,
        'skipped_counts': skipped_counts,
        'target_results': results
    }
    
    with open(Path(args.output_dir) / 'train_processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results

# Added functions for shell script integration
def check_memory_usage(memory_limit_gb):
    """Check if memory usage is approaching limit."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)
    return memory_gb, memory_gb > (memory_limit_gb * 0.9)

def cleanup_memory():
    """Force garbage collection to free memory."""
    gc.collect()

def update_progress(progress_file, target_id, status):
    """Update the progress tracking file."""
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "processed_targets": [],
                "failed_targets": [],
                "skipped_targets": []
            }
        
        # Update the appropriate list
        if status == "success":
            if target_id not in progress["processed_targets"]:
                progress["processed_targets"].append(target_id)
        elif status == "failure":
            if target_id not in progress["failed_targets"]:
                progress["failed_targets"].append(target_id)
        elif status == "skipped":
            if target_id not in progress["skipped_targets"]:
                progress["skipped_targets"].append(target_id)
        
        # Add latest update timestamp
        progress["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Write back to file
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
            
    except Exception as e:
        print(f"Error updating progress file: {e}")

def generate_html_report(output_dir):
    """Generate HTML report of processing results."""
    try:
        # Load the progress file
        progress_file = os.path.join(output_dir, "logs", "train_progress.json")
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        # Generate simple HTML report
        report_file = os.path.join(output_dir, "reports", "train_report.html")
        
        with open(report_file, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>train Feature Extraction Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .stat-box {{ 
                        background-color: #f8f9fa; 
                        border-radius: 5px; 
                        padding: 15px; 
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .success {{ color: #27ae60; }}
                    .failure {{ color: #e74c3c; }}
                    .skipped {{ color: #f39c12; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                </style>
            </head>
            <body>
                <h1>train Feature Extraction Report</h1>
                <p>Started: {progress.get('started', 'Unknown')}</p>
                <p>Last update: {progress.get('last_update', 'Unknown')}</p>
                
                <div class="stats">
                    <div class="stat-box">
                        <h2 class="success">{len(progress.get('processed_targets', []))}</h2>
                        <p>Processed</p>
                    </div>
                    <div class="stat-box">
                        <h2 class="failure">{len(progress.get('failed_targets', []))}</h2>
                        <p>Failed</p>
                    </div>
                    <div class="stat-box">
                        <h2 class="skipped">{len(progress.get('skipped_targets', []))}</h2>
                        <p>Skipped</p>
                    </div>
                </div>
                
                <h2>Failed Targets</h2>
                <table>
                    <tr>
                        <th>Target ID</th>
                    </tr>
                    {"".join(f"<tr><td>{target}</td></tr>" for target in progress.get('failed_targets', []))}
                </table>
                
                <h2>Processed Targets</h2>
                <table>
                    <tr>
                        <th>Target ID</th>
                    </tr>
                    {"".join(f"<tr><td>{target}</td></tr>" for target in progress.get('processed_targets', []))}
                </table>
            </body>
            </html>
            """)
        
        print(f"Generated HTML report at {report_file}")
        return report_file
    except Exception as e:
        print(f"Error generating report: {e}")
        return None

# Main function with shell script integration
def main():
    parser = argparse.ArgumentParser(description="Extract features for train data")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with sequences")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit processing to N targets")
    parser.add_argument("--cores", type=int, default=None, help="Number of cores to use")
    parser.add_argument("--pf-scale", type=float, default=1.5, help="Partition function scaling factor")
    parser.add_argument("--target", type=str, default="", help="Process only the specified target")
    parser.add_argument("--targets-file", type=str, default="", help="Process targets listed in file")
    parser.add_argument("--skip-existing", action="store_true", help="Skip targets with existing feature files")
    parser.add_argument("--memory-limit", type=float, default=8.0, help="Memory limit in GB")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--progress-file", type=str, default="", help="Progress tracking file")
    parser.add_argument("--processed-targets", type=str, default="", help="Already processed targets (comma-separated)")
    parser.add_argument("--failed-targets", type=str, default="", help="Already failed targets (comma-separated)")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    global args
    args = parser.parse_args()
    
    # Process args
    VERBOSE = args.verbose
    
    # Define data directories
    DATA_DIR = Path("data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = Path(args.output_dir)
    
    # Output directories for each feature type
    THERMO_DIR = PROCESSED_DIR / "thermo_features"
    MI_DIR = PROCESSED_DIR / "mi_features"
    DIHEDRAL_DIR = PROCESSED_DIR / "dihedral_features"
    
    # Make sure all directories exist
    for directory in [RAW_DIR, PROCESSED_DIR, THERMO_DIR, MI_DIR, DIHEDRAL_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Initialize progress tracking
    progress_file = args.progress_file
    processed_targets = []
    failed_targets = []
    
    if args.processed_targets:
        processed_targets = args.processed_targets.split(',')
    
    if args.failed_targets:
        failed_targets = args.failed_targets.split(',')
    
    # Load data
    df = load_rna_data(args.csv)
    if df is None:
        print(f"Error loading data from {args.csv}")
        return 1
    
    # Get target IDs
    id_col = next((col for col in ["target_id", "ID", "id"] if col in df.columns), None)
    if id_col is None:
        print("Error: Could not find ID column in data")
        return 1
    
    # Get target IDs - handle both formats
    if any("_" in str(id_val) for id_val in df[id_col] if isinstance(id_val, str)):
        # Contains underscores - might need to extract base IDs
        target_ids = get_unique_target_ids(df, id_col=id_col)
    else:
        # IDs are already in the right format
        target_ids = df[id_col].unique().tolist()
        print(f"Found {len(target_ids)} unique target IDs")
    
    # Handle single target option
    if args.target:
        if args.target in target_ids:
            target_ids = [args.target]
            print(f"Processing only target: {args.target}")
        else:
            print(f"Target {args.target} not found in available targets")
            return 1
    
    # Handle targets from file
    if args.targets_file and os.path.exists(args.targets_file):
        with open(args.targets_file, 'r') as f:
            file_targets = [line.strip() for line in f if line.strip()]
        
        # Filter to keep only targets that exist in data
        valid_targets = [t for t in file_targets if t in target_ids]
        if not valid_targets:
            print(f"Error: None of the targets in {args.targets_file} were found in the data")
            return 1
        
        target_ids = valid_targets
        print(f"Processing {len(target_ids)} targets from file: {args.targets_file}")
    
    # Apply limit if specified
    if args.limit > 0 and args.limit < len(target_ids):
        print(f"Limiting to first {args.limit} targets")
        target_ids = target_ids[:args.limit]
    
    # Filter out already processed targets
    if args.skip_existing or processed_targets:
        # First filter based on processed_targets list
        if processed_targets:
            target_ids = [t for t in target_ids if t not in processed_targets]
            print(f"Filtered out {len(processed_targets)} already processed targets")
        
        # Then check for existing files if skip-existing is set
        if args.skip_existing and not args.force:
            filtered_targets = []
            for target_id in target_ids:
                thermo_file = THERMO_DIR / f"{target_id}_thermo_features.npz"
                dihedral_file = DIHEDRAL_DIR / f"{target_id}_dihedral_features.npz"
                mi_file = MI_DIR / f"{target_id}_mi_features.npz"
                
                if thermo_file.exists() and dihedral_file.exists() and mi_file.exists():
                    print(f"Skipping {target_id} - feature files already exist")
                    update_progress(progress_file, target_id, "skipped")
                else:
                    filtered_targets.append(target_id)
                    
            print(f"Filtered out {len(target_ids) - len(filtered_targets)} targets with existing features")
            target_ids = filtered_targets
    
    if not target_ids:
        print("No targets to process after filtering")
        return 0
    
    # Process in batches to manage memory
    batch_size = args.batch_size
    memory_limit_gb = args.memory_limit
    
    print(f"Processing {len(target_ids)} targets in batches of {batch_size}")
    print(f"Memory limit: {memory_limit_gb} GB")
    
    # Process targets in batches
    total_targets = len(target_ids)
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    for i in range(0, total_targets, batch_size):
        batch_end = min(i + batch_size, total_targets)
        batch_targets = target_ids[i:batch_end]
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_targets + batch_size - 1)//batch_size}")
        print(f"Targets {i+1}-{batch_end} of {total_targets}")
        
        for idx, target_id in enumerate(batch_targets):
            print(f"\nProcessing target {i + idx + 1}/{total_targets}: {target_id}")
            
            # Check memory usage before processing
            memory_gb, approaching_limit = check_memory_usage(memory_limit_gb)
            if approaching_limit:
                print(f"WARNING: Approaching memory limit ({memory_gb:.2f} GB). Running cleanup...")
                cleanup_memory()
            
            # Skip if this target was previously processed or failed
            if target_id in processed_targets:
                print(f"Skipping {target_id} - already processed")
                skipped_count += 1
                update_progress(progress_file, target_id, "skipped")
                continue
            
            try:
                # Process the target
                result = process_target(
                    target_id,
                    extract_thermo=True,
                    extract_dihedral=True,
                    extract_mi=True
                )
                
                # Check result status
                all_success = True
                for feature_type in result.keys():
                    if feature_type == 'target_id' or feature_type == 'elapsed_time':
                        continue
                    if not result[feature_type].get('success', False):
                        all_success = False
                        break
                
                if all_success:
                    print(f"Successfully processed {target_id}")
                    success_count += 1
                    update_progress(progress_file, target_id, "success")
                else:
                    print(f"Failed to process some features for {target_id}")
                    failure_count += 1
                    update_progress(progress_file, target_id, "failure")
                
            except Exception as e:
                print(f"Error processing {target_id}: {e}")
                traceback.print_exc()
                failure_count += 1
                update_progress(progress_file, target_id, "failure")
        
        # Cleanup after batch
        print("Batch complete. Running memory cleanup...")
        cleanup_memory()
    
    # Generate report if requested
    if args.report:
        report_file = generate_html_report(args.output_dir)
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Total targets: {total_targets}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Skipped: {skipped_count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
PYTHONSCRIPT

# Make the script executable
chmod +x "$OUTPUT_DIR/logs/train_extraction_script.py"

# Prepare command line arguments for the Python script
SCRIPT_ARGS=""
SCRIPT_ARGS+=" --csv $CSV_FILE"
SCRIPT_ARGS+=" --output-dir $OUTPUT_DIR"
SCRIPT_ARGS+=" --memory-limit $MEMORY_LIMIT"
SCRIPT_ARGS+=" --batch-size $BATCH_SIZE"
SCRIPT_ARGS+=" --progress-file $PROGRESS_FILE"
SCRIPT_ARGS+=" --cores $CORES"
SCRIPT_ARGS+=" --pf-scale $PF_SCALE"

# Add processed and failed targets if resuming
if [[ "$RESUME" == "true" ]]; then
    PROCESSED_TARGETS_STR=$(IFS=,; echo "${PROCESSED_TARGETS[*]}")
    FAILED_TARGETS_STR=$(IFS=,; echo "${FAILED_TARGETS[*]}")
    
    SCRIPT_ARGS+=" --processed-targets $PROCESSED_TARGETS_STR"
    SCRIPT_ARGS+=" --failed-targets $FAILED_TARGETS_STR"
fi

# Add remaining options
[[ "$LIMIT" -gt 0 ]] && SCRIPT_ARGS+=" --limit $LIMIT"
[[ -n "$TARGET" ]] && SCRIPT_ARGS+=" --target $TARGET"
[[ -n "$TARGETS_FILE" ]] && SCRIPT_ARGS+=" --targets-file $TARGETS_FILE"
[[ "$SKIP_EXISTING" == "true" ]] && SCRIPT_ARGS+=" --skip-existing"
[[ "$VERBOSE" == "true" ]] && SCRIPT_ARGS+=" --verbose"
[[ "$FORCE" == "true" ]] && SCRIPT_ARGS+=" --force"
[[ "$REPORT" == "true" ]] && SCRIPT_ARGS+=" --report"

# Run the Python script
echo "Running train feature extraction..."
echo "python $OUTPUT_DIR/logs/train_extraction_script.py$SCRIPT_ARGS"

# Create a log file for the run
LOG_FILE="$OUTPUT_DIR/logs/train_extraction_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

# Execute the script
python "$OUTPUT_DIR/logs/train_extraction_script.py"$SCRIPT_ARGS 2>&1 | tee "$LOG_FILE"

# Check for success
RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "===== train Feature Extraction Complete ====="
    echo "Output saved to $OUTPUT_DIR"
    echo "Log saved to $LOG_FILE"
    
    # Generate final report if needed
    if [[ "$REPORT" == "true" ]]; then
        echo "Generating final HTML report..."
        python -c "
import json, os
from datetime import datetime

progress_file = '$PROGRESS_FILE'
output_dir = '$OUTPUT_DIR'
try:
    # Load progress data
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    # Compute statistics
    processed = len(progress.get('processed_targets', []))
    failed = len(progress.get('failed_targets', []))
    skipped = len(progress.get('skipped_targets', []))
    total = processed + failed
    success_rate = processed / total if total > 0 else 0
    
    # Generate simple HTML report
    report_file = os.path.join(output_dir, 'reports', 'train_final_report.html')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>train Feature Extraction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ 
                    background-color: #f8f9fa; 
                    border-radius: 5px; 
                    padding: 15px; 
                    text-align: center;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    min-width: 150px;
                }}
                .success {{ color: #27ae60; }}
                .failure {{ color: #e74c3c; }}
                .skipped {{ color: #f39c12; }}
                .highlight {{ font-size: 24px; font-weight: bold; }}
                progress {{ width: 100%; height: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>train Feature Extraction Final Report</h1>
            <p>Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Started: {progress.get('started', 'Unknown')}</p>
            
            <div class=\"stats\">
                <div class=\"stat-box\">
                    <p>Success Rate</p>
                    <p class=\"highlight\">{success_rate:.1%}</p>
                    <progress value=\"{success_rate}\" max=\"1\"></progress>
                </div>
                <div class=\"stat-box\">
                    <p>Processed</p>
                    <p class=\"highlight success\">{processed}</p>
                </div>
                <div class=\"stat-box\">
                    <p>Failed</p>
                    <p class=\"highlight failure\">{failed}</p>
                </div>
                <div class=\"stat-box\">
                    <p>Skipped</p>
                    <p class=\"highlight skipped\">{skipped}</p>
                </div>
            </div>
            
            <h2>Failed Targets</h2>
            <table>
                <tr>
                    <th>Target ID</th>
                </tr>
                {''.join(f'<tr><td>{target}</td></tr>' for target in progress.get('failed_targets', []))}
            </table>
            
            <h2>Skipped Targets</h2>
            <table>
                <tr>
                    <th>Target ID</th>
                </tr>
                {''.join(f'<tr><td>{target}</td></tr>' for target in progress.get('skipped_targets', []))}
            </table>
        </body>
        </html>
        ''')
    
    print(f'Generated final report at {report_file}')
except Exception as e:
    print(f'Error generating final report: {e}')
"
    fi
    
    # Verify feature compatibility
    echo "Verifying feature compatibility..."
    if [ -f "../scripts/verify_feature_compatibility.py" ]; then
        python ../scripts/verify_feature_compatibility.py "$OUTPUT_DIR" --verbose
    else
        echo "Warning: Feature verification script not found"
    fi
    
    exit 0
else
    echo "===== train Feature Extraction Failed ====="
    echo "Check $LOG_FILE for details"
    exit $RESULT
fi