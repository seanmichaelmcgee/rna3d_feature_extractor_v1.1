# %% [markdown]
# # RNA 3D Train Features Extraction
# 
# This notebook extracts all three types of features for RNA training data:
# 1. Thermodynamic features from RNA sequences
# 2. Pseudodihedral angle features from 3D coordinates
# 3. Mutual Information features from Multiple Sequence Alignments (MSAs)
# 
# This notebook works with training data that includes 3D structural information.
# 
# ## Dependencies
# - ViennaRNA (for thermodynamic features)
# - NumPy/SciPy/Pandas (core data processing)
# - Memory monitoring tools from src.analysis.memory_monitor
# - Feature extraction functions from src.analysis modules

# %%
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

# %% [markdown]
# # Configuration
# 
# # Define paths and parameters for feature extraction.

# %%
# Define relative paths
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Output directories for each feature type
THERMO_DIR = PROCESSED_DIR / "thermo_features"
DIHEDRAL_DIR = PROCESSED_DIR / "dihedral_features"
MI_DIR = PROCESSED_DIR / "mi_features"
MEMORY_PLOTS_DIR = PROCESSED_DIR / "memory_plots"

# Make sure all directories exist
for directory in [RAW_DIR, PROCESSED_DIR, THERMO_DIR, DIHEDRAL_DIR, MI_DIR, MEMORY_PLOTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)
            
# Parameters
LIMIT = 5  # Limit for testing; set to None to process all data
VERBOSE = True  # Whether to print detailed progress

# Auto-detect if running on Kaggle
KAGGLE_MODE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
if KAGGLE_MODE:
    print("Running in Kaggle environment")

# %% [markdown]
# # Helper Functions
# 
# # Define utility functions for loading data and extracting features.

# %%
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

def load_structure_data(target_id, data_dir=RAW_DIR):
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

def load_msa_data(target_id, data_dir=RAW_DIR):
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

def get_sequence_for_target(target_id, data_dir=RAW_DIR):
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

# %% [markdown]
# ## Feature Extraction Functions
# 
# Define functions for extracting each type of feature.

# %%
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
        output_file = THERMO_DIR / f"{target_id}_thermo_features.npz"
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
        output_file = DIHEDRAL_DIR / f"{target_id}_dihedral_features.npz"
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
            mi_result = calculate_mutual_information(msa_sequences, verbose=VERBOSE)
        
        if mi_result is None:
            print(f"Failed to calculate MI for {target_id}")
            return None
        
        # Convert to evolutionary features
        output_file = MI_DIR / f"{target_id}_mi_features.npz"
        
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

# %% [markdown]
# ## Batch Processing
# 
# Process multiple targets in batch mode.

# %%
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
        thermo_file = THERMO_DIR / f"{target_id}_thermo_features.npz"
        
        if thermo_file.exists():
            print(f"Thermodynamic features already exist for {target_id}")
            results['thermo'] = {'success': True, 'skipped': True}
        else:
            thermo_features = extract_thermo_features_for_target(target_id, sequence)
            results['thermo'] = {'success': thermo_features is not None}
    
    # Extract dihedral features
    if extract_dihedral:
        dihedral_file = DIHEDRAL_DIR / f"{target_id}_dihedral_features.npz"
        
        if dihedral_file.exists():
            print(f"Dihedral features already exist for {target_id}")
            results['dihedral'] = {'success': True, 'skipped': True}
        else:
            dihedral_features = extract_dihedral_features_for_target(target_id, structure_data)
            results['dihedral'] = {'success': dihedral_features is not None}
    
    # Extract MI features
    if extract_mi:
        mi_file = MI_DIR / f"{target_id}_mi_features.npz"
        
        if mi_file.exists():
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
    
    with open(PROCESSED_DIR / 'train_processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results

# %% [markdown]
# ## Load Data and Process

# %%
# Log initial memory usage for the entire run
log_memory_usage("Initial memory before loading data")

# Load training data - both labels and sequences
train_labels_file = RAW_DIR / "train_labels.csv"
train_sequences_file = RAW_DIR / "train_sequences.csv"

train_labels = load_rna_data(train_labels_file)
train_sequences = load_rna_data(train_sequences_file)

log_memory_usage("After loading training data")

if train_labels is None:
    print("Error loading training labels. Please make sure train_labels.csv exists.")
elif train_sequences is None:
    print("Error loading training sequences. Please make sure train_sequences.csv exists.")
else:
    # Get unique target IDs from labels
    target_ids = get_unique_target_ids(train_labels)
    
    # Verify sequences exist for target IDs
    seq_id_col = next((col for col in ["target_id", "ID", "id"] if col in train_sequences.columns), None)
    if seq_id_col:
        available_targets = set(train_sequences[seq_id_col])
        target_with_sequences = [tid for tid in target_ids if tid in available_targets]
        missing_sequences = len(target_ids) - len(target_with_sequences)
        
        if missing_sequences > 0:
            print(f"Warning: {missing_sequences} targets do not have sequences in train_sequences.csv")
        
        print(f"Found sequences for {len(target_with_sequences)}/{len(target_ids)} targets")
        target_ids = target_with_sequences
    
    # Limit for testing
    if LIMIT is not None and LIMIT < len(target_ids):
        print(f"Limiting to first {LIMIT} targets for testing")
        target_ids = target_ids[:LIMIT]
    
    # Process targets
    with MemoryTracker("Batch processing"):
        results = batch_process_targets(
            target_ids,
            extract_thermo=True,
            extract_dihedral=True,
            extract_mi=True
        )
    
    # Verify features
    print("\nVerifying processed features for compatibility...")
    import subprocess
    import sys
    
    verification_script = Path("../scripts/verify_feature_compatibility.py")
    if verification_script.exists():
        try:
            # Run the script as a subprocess
            cmd = [sys.executable, str(verification_script), str(PROCESSED_DIR), "--verbose"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print the output
            print(result.stdout)
            
            # Check for errors
            if result.returncode != 0:
                print(f"Verification failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
        except Exception as e:
            print(f"Error running verification script: {e}")
    else:
        print(f"Warning: Feature verification script not found at {verification_script}")
    
    # Plot memory usage
    log_memory_usage("Final memory usage")
    #plot_memory_usage(output_file=MEMORY_PLOTS_DIR / "train_memory_usage.png")

# %% [markdown]
# # Memory Profiling for Different RNA Lengths
# 
# This section includes memory profiling for different RNA sequence lengths to understand resource requirements.

# %%
# Function to create and profile sequences of different lengths
def profile_memory_for_lengths(lengths=[100, 500, 1000, 2000, 3000]):
    """Profile memory usage for different RNA sequence lengths."""
    from src.analysis.memory_monitor import profile_rna_length_memory
    
    # Reset memory history
    from src.analysis.memory_monitor import memory_history
    memory_history['timestamps'] = []
    memory_history['usage_gb'] = []
    memory_history['labels'] = []
    
    # Create output directory for memory plots
    memory_dir = PROCESSED_DIR / "memory_profiling"
    memory_dir.mkdir(exist_ok=True, parents=True)
    
    # Run profiling
    results = profile_rna_length_memory(
        seq_lengths=lengths,
        output_dir=memory_dir
    )
    
    # Create a summary table
    print("\nMemory Usage by Sequence Length:")
    print("-" * 60)
    print(f"{'Length (nt)':<15} {'Peak Memory (GB)':<20} {'Time (s)':<15}")
    print("-" * 60)
    for length in sorted(results.keys()):
        print(f"{length:<15} {results[length]['peak_memory_gb']:<20.2f} {results[length]['processing_time']:<15.2f}")
    
    return results

# Uncomment and run this cell to profile memory usage for different sequence lengths
# This might take some time to complete
# profile_results = profile_memory_for_lengths(lengths=[100, 500, 1000, 2000, 3000])

# %% [markdown]
# # Data Loader Compatibility Check
# 
# Let's verify that our features are compatible with the downstream data loader requirements.

# %%
# Run the feature verification script on our extracted features
import subprocess
import sys

def verify_features():
    """Run the feature verification script on the processed data directory."""
    verification_script = Path("../scripts/verify_feature_compatibility.py")
    
    # Check if the script exists
    if not verification_script.exists():
        print(f"Error: Verification script not found at {verification_script}")
        return False
    
    print(f"Running feature verification script on {PROCESSED_DIR}")
    try:
        # Run the script as a subprocess
        cmd = [sys.executable, str(verification_script), str(PROCESSED_DIR), "--verbose"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print the output
        print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print(f"Verification failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"Error running verification script: {e}")
        return False

# Uncomment to run the verification when features are available
compatibility_check = verify_features()

# %%
# Function to create and profile sequences of different lengths
def profile_memory_for_lengths(lengths=[100, 500, 1000, 2000, 3000]):
    """Profile memory usage for different RNA sequence lengths."""
    from src.analysis.memory_monitor import profile_rna_length_memory
    
    # Reset memory history
    from src.analysis.memory_monitor import memory_history
    memory_history['timestamps'] = []
    memory_history['usage_gb'] = []
    memory_history['labels'] = []
    
    # Create output directory for memory plots
    memory_dir = PROCESSED_DIR / "memory_profiling"
    memory_dir.mkdir(exist_ok=True, parents=True)
    
    # Run profiling
    results = profile_rna_length_memory(
        seq_lengths=lengths,
        output_dir=memory_dir
    )
    
    # Create a summary table
    print("\nMemory Usage by Sequence Length:")
    print("-" * 60)
    print(f"{'Length (nt)':<15} {'Peak Memory (GB)':<20} {'Time (s)':<15}")
    print("-" * 60)
    for length in sorted(results.keys()):
        print(f"{length:<15} {results[length]['peak_memory_gb']:<20.2f} {results[length]['processing_time']:<15.2f}")
    
    return results

# Uncomment and run this cell to profile memory usage for different sequence lengths
# This might take some time to complete
# profile_results = profile_memory_for_lengths(lengths=[100, 500, 1000, 2000, 3000])


