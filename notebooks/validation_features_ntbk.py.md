# %% [markdown]
# # RNA 3D Validation Features Extraction
# 
# This notebook extracts all three types of features for RNA validation data:
# 1. Thermodynamic features from RNA sequences
# 2. Pseudodihedral angle features from 3D coordinates
# 3. Mutual Information features from Multiple Sequence Alignments (MSAs)
# 
# This notebook works with validation data that includes 3D structural information.
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

# Ensure the parent directory is in the path so we can import our module
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
    # Extract just the base target ID (e.g., "R1149" from "R1149_62")
    target_ids = []
    for id_str in df[id_col]:
        base_id = id_str.split('_')[0]
        target_ids.append(base_id)
    
    unique_targets = sorted(list(set(target_ids)))
    print(f"Found {len(unique_targets)} unique target IDs")
    return unique_targets

def load_structure_data(target_id, data_dir=RAW_DIR):
    """
    Load structure data for a given target from labels CSV.
    """
    data_dir = Path(data_dir)
    
    # Define possible label files
    label_files = [
        data_dir / "validation_labels.csv",
        data_dir / "test_labels.csv"
    ]
    
    for label_file in label_files:
        if label_file.exists():
            try:
                print(f"Looking for structure data for {target_id} in {label_file}")
                # Read the entire CSV file
                all_data = pd.read_csv(label_file)
                
                # For validation data, match rows where ID starts with target_id_
                if "ID" in all_data.columns:
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
    # Extract base ID without residue number
    base_id = target_id.split('_')[0]
    
    # Define possible MSA directories and extensions
    msa_dirs = [
        data_dir / "MSA",
        data_dir,
        data_dir / "alignments",
        data_dir / "msa",
        data_dir / "validation" / "MSA",
        data_dir / "validation"
    ]
    
    extensions = [".MSA.fasta", ".fasta", ".fa", ".afa", ".msa"]
    
    # Try with base ID and original target ID
    for id_to_try in [base_id, target_id]:
        for msa_dir in msa_dirs:
            if not msa_dir.exists():
                continue
                
            for ext in extensions:
                msa_path = msa_dir / f"{id_to_try}{ext}"
                if msa_path.exists():
                    print(f"Loading MSA data from {msa_path}")
                    try:
                        # Parse FASTA file
                        sequences = []
                        current_seq = ""
                        
                        with open(msa_path, 'r') as f:
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
    
    # Last resort: recursive search for files containing the base ID
    try:
        for pattern in [f"**/{base_id}*.fasta", f"**/{base_id}*.fa", f"**/{base_id}*.msa"]:
            matches = list(data_dir.glob(pattern))
            if matches:
                msa_path = matches[0]
                print(f"Found MSA via recursive search: {msa_path}")
                
                # Parse the file
                sequences = []
                current_seq = ""
                
                with open(msa_path, 'r') as f:
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
        print(f"Error in recursive MSA search: {e}")
    
    print(f"Could not find MSA data for {target_id}")
    return None

def get_sequence_for_target(target_id, data_dir=RAW_DIR):
    """
    Get RNA sequence for a target ID from the sequence file.
    
    Args:
        target_id: Target ID (e.g., "R1107_A")
        data_dir: Directory containing sequence data
        
    Returns:
        RNA sequence as string or None if not found
    """
    # Try validation_sequences.csv first
    validation_seq_path = data_dir / "validation_sequences.csv"
    if validation_seq_path.exists():
        try:
            df = pd.read_csv(validation_seq_path)
            # Try different possible column names for ID and sequence
            id_cols = ["target_id", "ID", "id"]
            seq_cols = ["sequence", "Sequence", "seq"]
            
            for id_col in id_cols:
                if id_col in df.columns:
                    for seq_col in seq_cols:
                        if seq_col in df.columns:
                            # Try exact match first
                            target_row = df[df[id_col] == target_id]
                            if len(target_row) > 0:
                                sequence = target_row[seq_col].iloc[0]
                                print(f"Found sequence for {target_id} in validation_sequences.csv (exact match), length: {len(sequence)}")
                                return sequence
                            
                            # If exact match fails, try base target ID (remove residue number)
                            base_id = target_id.split('_')[0]
                            if '_' in target_id:
                                # Try with just first component
                                target_row = df[df[id_col] == base_id]
                                if len(target_row) > 0:
                                    sequence = target_row[seq_col].iloc[0]
                                    print(f"Found sequence for {target_id} using base ID {base_id}, length: {len(sequence)}")
                                    return sequence
                                
                                # Try with first two components (target and chain)
                                parts = target_id.split('_')
                                if len(parts) >= 2:
                                    base_id_with_chain = f"{parts[0]}_{parts[1]}"
                                    target_row = df[df[id_col] == base_id_with_chain]
                                    if len(target_row) > 0:
                                        sequence = target_row[seq_col].iloc[0]
                                        print(f"Found sequence for {target_id} using base+chain ID {base_id_with_chain}, length: {len(sequence)}")
                                        return sequence
                            
                            # Try partial matching
                            for row_id in df[id_col]:
                                if str(row_id) in target_id or target_id.startswith(str(row_id)):
                                    target_row = df[df[id_col] == row_id]
                                    sequence = target_row[seq_col].iloc[0]
                                    print(f"Found sequence for {target_id} using partial match with {row_id}, length: {len(sequence)}")
                                    return sequence
        except Exception as e:
            print(f"Error loading sequence data from {validation_seq_path}: {e}")
    
    # Continue with the rest of your existing function...
    
    # If not found in validation_sequences.csv, try other sequence files
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
# # Feature Extraction Functions
# 
# # Define functions for extracting each type of feature.

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

# %%
def extract_dihedral_features_for_all_structures(target_id, structure_data=None):
    """
    Extract pseudodihedral angle features for all structure sets in validation data.
    
    Args:
        target_id: Target ID
        structure_data: DataFrame with structure coordinates
        
    Returns:
        Dictionary with dihedral features for all structures or None if failed
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
            print(f"Not enough residues ({len(structure_data)}) for {target_id}")
            return None
        
        # Find all coordinate sets (x_1, y_1, z_1), (x_2, y_2, z_2), etc.
        x_cols = sorted([col for col in structure_data.columns if col.startswith('x_')])
        y_cols = sorted([col for col in structure_data.columns if col.startswith('y_')])
        z_cols = sorted([col for col in structure_data.columns if col.startswith('z_')])
        
        # Make sure we have matching coordinate sets
        num_structures = len(x_cols)
        if not (len(x_cols) == len(y_cols) == len(z_cols)):
            print(f"Mismatched coordinate columns: {len(x_cols)} x-cols, {len(y_cols)} y-cols, {len(z_cols)} z-cols")
            return None
            
        print(f"Found {num_structures} structure sets for {target_id}")
        
        # Output file path
        output_file = DIHEDRAL_DIR / f"{target_id}_dihedral_features.npz"
        
        # Process each structure set
        all_features = {
            'target_id': target_id,
            'num_structures': num_structures,
            'structure_ids': list(range(1, num_structures + 1))
        }
        
        for i in range(num_structures):
            # Get column names for this structure
            struct_idx = i + 1  # 1-based indexing in column names
            x_col = f'x_{struct_idx}'
            y_col = f'y_{struct_idx}'
            z_col = f'z_{struct_idx}'
            
            # Skip if any column is missing
            if not all(col in structure_data.columns for col in [x_col, y_col, z_col]):
                print(f"Skipping structure {struct_idx} due to missing coordinates")
                continue
            
            # Create a copy with renamed columns for compatibility with dihedral_analysis
            struct_data = structure_data.copy()
            
            # Create 'resid' column if not present
            if 'resid' not in struct_data.columns:
                if 'residue' in struct_data.columns:
                    struct_data['resid'] = struct_data['residue']
                else:
                    struct_data['resid'] = list(range(1, len(struct_data) + 1))
            
            # Rename coordinate columns to the standard names expected by dihedral_analysis
            struct_data = struct_data.rename(columns={
                x_col: 'x_1',
                y_col: 'y_1',
                z_col: 'z_1'
            })
            
            print(f"Processing structure {struct_idx}/{num_structures}")
            
            # Calculate dihedral features for this structure
            try:
                dihedral_features = extract_dihedral_features(
                    struct_data, 
                    output_file=None,  # Don't save individual structures
                    include_raw_angles=True
                )
                
                if dihedral_features is not None:
                    # Add prefixed keys to identify this structure set
                    for key, value in dihedral_features.items():
                        all_features[f'struct_{struct_idx}_{key}'] = value
                    
                    print(f"✅ Successfully processed structure {struct_idx}")
                else:
                    print(f"❌ Failed to extract features for structure {struct_idx}")
                    
            except Exception as e:
                print(f"Error processing structure {struct_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Check if we have at least one successful structure
        successful_structs = sum(1 for key in all_features.keys() if key.startswith('struct_'))
        if successful_structs == 0:
            print(f"No structures were successfully processed for {target_id}")
            return None
            
        # Save combined features
        print(f"Saving combined features for {successful_structs}/{num_structures} structures")
        save_features_npz(all_features, output_file)
        
        elapsed_time = time.time() - start_time
        print(f"Completed dihedral extraction in {elapsed_time:.2f} seconds")
        
        return all_features
        
    except Exception as e:
        print(f"Error in dihedral extraction for {target_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

# %%

def extract_dihedral_features_for_target(target_id, structure_data=None):
    """
    Extract pseudodihedral angle features for a given target with multiple coordinate sets.
    
    Args:
        target_id: Target ID
        structure_data: DataFrame with structure coordinates (optional)
        
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
        
        # Find coordinate sets
        x_cols = sorted([col for col in structure_data.columns if col.startswith('x_')])
        
        # Check how many residues we have
        num_residues = len(structure_data)
        print(f"Found {num_residues} residues and {len(x_cols)} coordinate sets")
        
        if num_residues < 4:
            print(f"Not enough residues ({num_residues}) for {target_id}")
            return None
            
        # Initialize result dictionary
        result_features = {
            'target_id': target_id,
            'num_structures': 0,
            'structure_ids': []
        }
        
        # Process each coordinate set
        for coord_set in range(1, len(x_cols) + 1):
            x_col = f'x_{coord_set}'
            y_col = f'y_{coord_set}'
            z_col = f'z_{coord_set}'
            
            # Skip if any of these columns doesn't exist
            if not all(col in structure_data.columns for col in [x_col, y_col, z_col]):
                continue
            
            # Check if there are valid coordinates (not -1e+18)
            has_valid_coords = False
            for _, row in structure_data.iterrows():
                if (abs(row[x_col]) < 1e17 and 
                    abs(row[y_col]) < 1e17 and 
                    abs(row[z_col]) < 1e17):
                    has_valid_coords = True
                    break
                    
            if not has_valid_coords:
                print(f"Skipping coordinate set {coord_set} (no valid coordinates)")
                continue
                
            print(f"Processing coordinate set {coord_set}")
            
            # Create a fresh DataFrame with only the necessary columns
            # This avoids the DataFrame alignment issues
            filtered_rows = []
            for idx, row in structure_data.iterrows():
                if (abs(row[x_col]) < 1e17 and 
                    abs(row[y_col]) < 1e17 and 
                    abs(row[z_col]) < 1e17):
                    # Keep only needed columns
                    filtered_row = {
                        'resid': row.get('resid', row.get('residue', idx+1)),
                        'x_1': row[x_col],
                        'y_1': row[y_col],
                        'z_1': row[z_col]
                    }
                    # Add any other required columns
                    if 'resname' in structure_data.columns:
                        filtered_row['resname'] = row['resname']
                    filtered_rows.append(filtered_row)
            
            # Create a new DataFrame from the filtered rows
            filtered_df = pd.DataFrame(filtered_rows)
            
            # Make sure we have enough residues after filtering
            if len(filtered_df) < 4:
                print(f"Not enough valid residues in coordinate set {coord_set} after filtering ({len(filtered_df)})")
                continue
            
            print(f"Using {len(filtered_df)}/{len(structure_data)} residues with valid coordinates")
            
            # Calculate dihedral features
            try:
                dihedral_features = extract_dihedral_features(
                    filtered_df,
                    output_file=None,
                    include_raw_angles=True
                )
                
                if dihedral_features is not None:
                    # Store features with a prefix indicating the coordinate set
                    for key, value in dihedral_features.items():
                        result_features[f'struct_{coord_set}_{key}'] = value
                    
                    # Update metadata
                    result_features['num_structures'] += 1
                    result_features['structure_ids'].append(coord_set)
                    
                    print(f"✅ Successfully processed coordinate set {coord_set}")
                else:
                    print(f"❌ Failed to extract dihedral features for coordinate set {coord_set}")
            except Exception as e:
                print(f"Error processing coordinate set {coord_set}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save output
        if result_features['num_structures'] > 0:
            output_file = DIHEDRAL_DIR / f"{target_id}_dihedral_features.npz"
            save_features_npz(result_features, output_file)
            print(f"Saved dihedral features for {result_features['num_structures']} coordinate sets")
            return result_features
        else:
            print(f"No valid coordinate sets found for {target_id}")
            return None
            
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

# %%
def test_validation_dihedral_extraction(test_target="R1107"):  # Use a default test target
    """Test dihedral extraction on a single validation target."""
    print(f"Testing dihedral extraction for validation target: {test_target}")
    
    # Load structure data
    structure_data = load_structure_data(test_target)
    if structure_data is None:
        print(f"❌ Failed to load structure data for {test_target}")
        return
    
    # Count coordinate sets
    x_cols = [col for col in structure_data.columns if col.startswith('x_')]
    print(f"Found {len(x_cols)} coordinate sets: {', '.join(x_cols[:5])}...")
    
    # Extract dihedral features
    features = extract_dihedral_features_for_target(test_target, structure_data)
    
    if features is not None:
        print(f"✅ Successfully extracted dihedral features")
        print(f"Number of structures: {features.get('num_structures', 'Unknown')}")
        print("Feature keys examples:")
        for key in list(features.keys())[:5]:
            print(f"  - {key}")
    else:
        print(f"❌ Failed to extract dihedral features")

# Run the test with a specific RNA target ID from the validation set
test_validation_dihedral_extraction("R1107")

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
    
    with open(PROCESSED_DIR / 'validation_processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results

# %% [markdown]
# ## Load Data and Process

# %%
# Log initial memory usage for the entire run
log_memory_usage("Initial memory before loading data")

# Load validationing data - both labels and sequences
validation_labels_file = RAW_DIR / "validation_labels.csv"
validation_sequences_file = RAW_DIR / "validation_sequences.csv"

validation_labels = load_rna_data(validation_labels_file)
validation_sequences = load_rna_data(validation_sequences_file)

log_memory_usage("After loading validationing data")

if validation_labels is None:
    print("Error loading validationing labels. Please make sure validation_labels.csv exists.")
elif validation_sequences is None:
    print("Error loading validationing sequences. Please make sure validation_sequences.csv exists.")
else:
    # Get unique target IDs from labels
    target_ids = get_unique_target_ids(validation_labels)
    
    # Verify sequences exist for target IDs
    seq_id_col = next((col for col in ["target_id", "ID", "id"] if col in validation_sequences.columns), None)
    if seq_id_col:
        # Check if IDs in sequence file contain underscores (indicating chain/structure info)
        sequence_ids = validation_sequences[seq_id_col].astype(str).tolist()
        
        if any("_" in str(id_val) for id_val in sequence_ids):
            # Sequence IDs have same format (with underscore), do direct comparison
            available_targets = set(sequence_ids)
            target_with_sequences = [tid for tid in target_ids if tid in available_targets]
        else:
            # Sequence IDs are in a different format, try to match the base part
            # For example, match "R1107" from sequences file with "R1107_A" from labels
            available_targets = set()
            for seq_id in sequence_ids:
                # Add the ID as is and also try adding common chain identifiers
                available_targets.add(seq_id)
                available_targets.add(f"{seq_id}_A")  # Common chain identifier
            
            target_with_sequences = [tid for tid in target_ids if tid in available_targets]
            
            # If still no matches, try more flexible matching
            if not target_with_sequences:
                # Try to match beginning of target ID with sequence ID
                target_with_sequences = []
                for tid in target_ids:
                    for seq_id in sequence_ids:
                        if seq_id in tid or tid.startswith(seq_id):
                            target_with_sequences.append(tid)
                            break
        
        missing_sequences = len(target_ids) - len(target_with_sequences)
        
        if missing_sequences > 0:
            print(f"Warning: {missing_sequences} targets do not have sequences in validation_sequences.csv")
        
        print(f"Found sequences for {len(target_with_sequences)}/{len(target_ids)} targets")
        target_ids = target_with_sequences
    
    # Limit for testing - add back in for regular workflow
    if LIMIT is not None and LIMIT < len(target_ids):
        print(f"Limiting to first {LIMIT} targets for testing")
       target_ids = target_ids[:LIMIT]
    
        # Simple hardcoded target selection
    # target_of_interest = "R1149"  # Replace with your target ID
    # selected_targets = [tid for tid in target_ids if tid.startswith(target_of_interest)]

    print(f"Selected {len(selected_targets)} targets that start with '{target_of_interest}':")
    for tid in selected_targets:
        print(f"  - {tid}")

    # Use these selected targets instead of the original list
    target_ids = selected_targets


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
    #plot_memory_usage(output_file=MEMORY_PLOTS_DIR / "validation_memory_usage.png")

# %%
def test_validation_dihedral_extraction():
    """Test dihedral extraction on a single validation target."""
    if not target_ids or len(target_ids) == 0:
        print("No target IDs available to test")
        return
    
    test_target = target_ids[0]
    print(f"Testing dihedral extraction for validation target: {test_target}")
    
    # Load structure data
    structure_data = load_structure_data(test_target)
    if structure_data is None:
        print(f"❌ Failed to load structure data for {test_target}")
        return
    
    # Count coordinate sets
    x_cols = [col for col in structure_data.columns if col.startswith('x_')]
    print(f"Found {len(x_cols)} coordinate sets: {', '.join(x_cols)}")
    
    # Extract dihedral features
    features = extract_dihedral_features_for_target(test_target, structure_data)
    
    if features is not None:
        print(f"✅ Successfully extracted dihedral features")
        print(f"Number of structures: {features.get('num_structures', 'Unknown')}")
        print("Feature keys:")
        for key in list(features.keys())[:10]:  # Show first 10 keys
            print(f"  - {key}")
        if len(features) > 10:
            print(f"  ... and {len(features) - 10} more")
    else: 
        print(f"❌ Failed to extract dihedral features")

# Run the test on a single target before batch processing
test_validation_dihedral_extraction()

# %% [markdown]
# # Data Loader Compatibility Check
# 
# Visualize and validate the features.

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
def debug_structure_data(target_id, data_dir=RAW_DIR):
    """Debug helper to diagnose structure loading issues."""
    print(f"\n=== DEBUGGING STRUCTURE DATA FOR {target_id} ===")
    
    # Try to load the data
    structure_data = load_structure_data(target_id, data_dir)
    
    if structure_data is None:
        print("❌ No structure data found!")
        
        # Check what label files exist
        label_files = [
            data_dir / "validation_labels.csv",
            data_dir / "test_labels.csv",
            data_dir / "train_labels.csv" 
        ]
        
        for label_file in label_files:
            if label_file.exists():
                print(f"📄 Found label file: {label_file}")
                
                # Check structure of file
                try:
                    df = pd.read_csv(label_file)
                    print(f"  - Columns: {df.columns.tolist()}")
                    print(f"  - Sample IDs: {df['ID'].head(3).tolist() if 'ID' in df.columns else 'No ID column'}")
                    
                    # Check if target ID exists in any similar form
                    base_id = target_id.split('_')[0]
                    matches = df[df['ID'].str.contains(base_id, regex=False)] if 'ID' in df.columns else None
                    if matches is not None and len(matches) > 0:
                        print(f"  - Found {len(matches)} rows containing '{base_id}'")
                        print(f"  - Sample IDs: {matches['ID'].head(3).tolist()}")
                except Exception as e:
                    print(f"  - Error reading file: {e}")
    else:
        print(f"✅ Found {len(structure_data)} residues for {target_id}")
        print(f"Columns: {structure_data.columns.tolist()}")
        
        # Check for coordinate columns
        x_cols = [col for col in structure_data.columns if col.startswith('x_')]
        y_cols = [col for col in structure_data.columns if col.startswith('y_')]
        z_cols = [col for col in structure_data.columns if col.startswith('z_')]
        
        print(f"Coordinate columns:")
        print(f"  - X columns: {x_cols}")
        print(f"  - Y columns: {y_cols}")
        print(f"  - Z columns: {z_cols}")
        
        # Check if proper coordinate columns exist
        if not (x_cols and y_cols and z_cols):
            print("❌ Missing coordinate columns!")
            
        # Print the first few rows for inspection
        print("\nFirst few rows:")
        print(structure_data.head(2))
    
    print("=" * 50)
    return structure_data


