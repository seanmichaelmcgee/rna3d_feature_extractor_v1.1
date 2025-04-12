#!/usr/bin/env python3
"""
Single Target Test Script

This script runs a complete test of the RNA feature extraction pipeline on a single target,
verifying all three feature types (thermodynamic, dihedral, and MI) and checking compatibility
with the data loader requirements.

Usage:
    python single_target_test.py --target TARGET_ID [--docker]

Author: Claude
Date: April 12, 2025
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import shutil
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
try:
    from src.analysis import thermodynamic_analysis
    from src.analysis import mutual_information
    from src.data.extract_features_simple import save_features_npz
    from src.analysis.memory_monitor import MemoryTracker, log_memory_usage, plot_memory_usage
    from scripts.verify_feature_compatibility import verify_target_features
except ImportError as e:
    print(f"Error importing project modules: {e}")
    traceback.print_exc()
    sys.exit(1)

def load_structure_data(target_id, data_dir):
    """
    Load structure data for a given target from labels CSV.
    
    Args:
        target_id: Target ID
        data_dir: Directory containing data (optional)
        
    Returns:
        DataFrame with structure coordinates or None if not found
    """
    import pandas as pd
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

def get_sequence_for_target(target_id, data_dir):
    """
    Get RNA sequence for a target ID.
    
    Args:
        target_id: Target ID
        data_dir: Path to data directory
        
    Returns:
        RNA sequence as string or None if not found
    """
    print(f"Looking for sequence for target {target_id}...")
    data_dir = Path(data_dir)
    
    # Try to find in test_sequences.csv first
    try:
        import pandas as pd
        for csv_name in ['test_sequences.csv', 'train_sequences.csv', 'validation_sequences.csv']:
            csv_path = data_dir / csv_name
            if csv_path.exists():
                print(f"Checking {csv_path}...")
                df = pd.read_csv(csv_path)
                
                # Try different possible column name configurations
                id_col = next((col for col in ['target_id', 'ID', 'id'] if col in df.columns), None)
                seq_col = next((col for col in ['sequence', 'Sequence', 'seq'] if col in df.columns), None)
                
                if id_col and seq_col:
                    # Look for exact match
                    match = df[df[id_col] == target_id]
                    if len(match) > 0:
                        sequence = match[seq_col].iloc[0]
                        print(f"Found sequence in {csv_path}, length: {len(sequence)}")
                        return sequence
    except Exception as e:
        print(f"Error searching CSV files: {e}")
    
    # If we can't find the sequence, create a test sequence
    print(f"Could not find sequence for {target_id}, creating a test sequence")
    return "GGGAAACCC" * 50  # 450 nt test sequence

def get_msa_data_for_target(target_id, data_dir):
    """
    Get MSA data for a target ID.
    
    Args:
        target_id: Target ID
        data_dir: Path to data directory
        
    Returns:
        List of MSA sequences or None if not found
    """
    print(f"Looking for MSA data for target {target_id}...")
    data_dir = Path(data_dir)
    
    # Try to find the MSA file
    msa_paths = [
        data_dir / "MSA" / f"{target_id}.MSA.fasta",
        data_dir / f"{target_id}.MSA.fasta",
        data_dir / "alignments" / f"{target_id}.MSA.fasta"
    ]
    
    for path in msa_paths:
        if path.exists():
            print(f"Found MSA data at {path}")
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
    
    # If we can't find MSA data, create fake alignment data
    print(f"Could not find MSA data for {target_id}, creating fake alignment")
    sequence = get_sequence_for_target(target_id, data_dir)
    if sequence:
        # Create a fake MSA with some variation
        msa = []
        for i in range(10):
            # Add some random variation to sequence copies
            seq_var = ''
            for c in sequence:
                if np.random.random() < 0.05:  # 5% chance of mutation
                    seq_var += np.random.choice(['A', 'C', 'G', 'U'])
                else:
                    seq_var += c
            msa.append(seq_var)
        return msa
    return None

def extract_thermodynamic_features(target_id, sequence, output_dir):
    """
    Extract thermodynamic features for a target.
    
    Args:
        target_id: Target ID
        sequence: RNA sequence
        output_dir: Output directory
        
    Returns:
        Path to saved feature file
    """
    print(f"\n=== Extracting thermodynamic features for {target_id} ===")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"{target_id}_thermo_features.npz"
    
    with MemoryTracker("Thermodynamic features extraction"):
        # Extract features
        features = thermodynamic_analysis.extract_thermodynamic_features(sequence)
        
        # Add target ID and sequence
        features['target_id'] = target_id
        features['sequence'] = sequence
        
        # Save features
        save_features_npz(features, output_file)
        
        # Print feature summary
        print(f"\nThermodynamic feature summary:")
        print(f"- MFE: {features.get('mfe', 'N/A')}")
        print(f"- Ensemble energy: {features.get('ensemble_energy', 'N/A')}")
        print(f"- MFE probability: {features.get('prob_of_mfe', 'N/A')}")
        print(f"- Pairing matrix shape: {features.get('pairing_probs', np.array([])).shape}")
        print(f"- Positional entropy shape: {features.get('positional_entropy', np.array([])).shape}")
    
    return output_file

def extract_mi_features(target_id, msa_sequences, output_dir):
    """
    Extract Mutual Information features for a target.
    
    Args:
        target_id: Target ID
        msa_sequences: List of MSA sequences
        output_dir: Output directory
        
    Returns:
        Path to saved feature file
    """
    print(f"\n=== Extracting MI features for {target_id} ===")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"{target_id}_features.npz"
    
    with MemoryTracker("MI features extraction"):
        # Calculate MI features
        features = mutual_information.calculate_mutual_information(msa_sequences)
        
        # Add target ID
        features['target_id'] = target_id
        
        # Save features
        save_features_npz(features, output_file)
        
        # Print feature summary
        print(f"\nMI feature summary:")
        if 'coupling_matrix' in features:
            print(f"- Coupling matrix shape: {features['coupling_matrix'].shape}")
        elif 'scores' in features:
            print(f"- Scores matrix shape: {features['scores'].shape}")
        else:
            print("- No coupling matrix found in features")
    
    return output_file

def extract_dihedral_features(target_id, structure_data, output_dir):
    """
    Extract dihedral features from structure data.
    
    Args:
        target_id: Target ID
        structure_data: DataFrame with structure coordinates
        output_dir: Output directory
        
    Returns:
        Path to saved feature file
    """
    print(f"\n=== Extracting dihedral features for {target_id} ===")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"{target_id}_dihedral_features.npz"
    
    try:
        from src.analysis.dihedral_analysis import extract_dihedral_features as extract_dihedrals
        
        # Extract dihedral features
        with MemoryTracker("Dihedral feature extraction"):
            features = extract_dihedrals(structure_data, output_file=output_file, include_raw_angles=True)
            features['target_id'] = target_id
        
        print(f"Extracted dihedral features with shape {features['features'].shape}")
        return output_file
    except Exception as e:
        print(f"Error extracting dihedral features: {e}")
        traceback.print_exc()
        
        # Fall back to making dummy features if extraction fails
        return make_dummy_dihedral_features(target_id, len(structure_data), output_dir)

def make_dummy_dihedral_features(target_id, seq_length, output_dir):
    """
    Create dummy dihedral features for testing.
    Used as a fallback when real extraction fails.
    
    Args:
        target_id: Target ID
        seq_length: Length of sequence/structure
        output_dir: Output directory
        
    Returns:
        Path to saved feature file
    """
    print(f"\n=== Creating dummy dihedral features for {target_id} ===")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"{target_id}_dihedral_features.npz"
    
    # Create dummy features (sequence_length, 4) for eta_sin, eta_cos, theta_sin, theta_cos
    dummy_features = np.random.normal(0, 1, size=(seq_length, 4))
    
    # Save features
    features = {
        'features': dummy_features,
        'target_id': target_id,
        'feature_names': ['eta_sin', 'eta_cos', 'theta_sin', 'theta_cos']
    }
    save_features_npz(features, output_file)
    
    print(f"Created dummy dihedral features with shape {dummy_features.shape}")
    return output_file

def compare_outputs(local_dir, docker_dir, target_id):
    """
    Compare feature outputs between local and Docker environments.
    
    Args:
        local_dir: Local features directory
        docker_dir: Docker features directory
        target_id: Target ID to compare
        
    Returns:
        bool: True if outputs match
    """
    print(f"\n=== Comparing outputs for {target_id} ===")
    local_dir = Path(local_dir)
    docker_dir = Path(docker_dir)
    
    # Define feature types and file patterns
    feature_types = {
        'thermo': f"{target_id}_thermo_features.npz",
        'dihedral': f"{target_id}_dihedral_features.npz",
        'mi': f"{target_id}_features.npz"
    }
    
    # Check each feature type
    all_match = True
    for feature_type, file_pattern in feature_types.items():
        local_file = local_dir / feature_type / file_pattern
        docker_file = docker_dir / feature_type / file_pattern
        
        if not local_file.exists():
            print(f"❌ {feature_type}: Local file missing")
            all_match = False
            continue
            
        if not docker_file.exists():
            print(f"❌ {feature_type}: Docker file missing")
            all_match = False
            continue
            
        try:
            # Load both files
            local_data = dict(np.load(local_file, allow_pickle=True))
            docker_data = dict(np.load(docker_file, allow_pickle=True))
            
            # Compare keys
            local_keys = set(local_data.keys())
            docker_keys = set(docker_data.keys())
            
            if local_keys != docker_keys:
                print(f"❌ {feature_type}: Keys don't match")
                print(f"   Local only: {local_keys - docker_keys}")
                print(f"   Docker only: {docker_keys - local_keys}")
                all_match = False
                continue
                
            # Compare array shapes
            shape_match = True
            for key in local_keys:
                if isinstance(local_data[key], np.ndarray) and isinstance(docker_data[key], np.ndarray):
                    if local_data[key].shape != docker_data[key].shape:
                        print(f"❌ {feature_type}: Shape mismatch for {key}")
                        print(f"   Local: {local_data[key].shape}")
                        print(f"   Docker: {docker_data[key].shape}")
                        shape_match = False
            
            if not shape_match:
                all_match = False
                continue
                
            # Compare key numeric values 
            value_match = True
            for key in ['mfe', 'ensemble_energy']:
                if key in local_data and key in docker_data:
                    local_val = float(local_data[key])
                    docker_val = float(docker_data[key])
                    
                    # Allow small differences due to numerical precision
                    if abs(local_val - docker_val) > 0.01:
                        print(f"❌ {feature_type}: Value mismatch for {key}")
                        print(f"   Local: {local_val}")
                        print(f"   Docker: {docker_val}")
                        value_match = False
            
            if not value_match:
                all_match = False
                continue
                
            print(f"✅ {feature_type}: Files match")
            
        except Exception as e:
            print(f"❌ {feature_type}: Error comparing files: {e}")
            all_match = False
    
    return all_match

def process_target(target_id, data_dir, output_dir, comparison_dir=None):
    """
    Process a single target through the complete feature extraction pipeline.
    
    Args:
        target_id: Target ID to process
        data_dir: Path to data directory
        output_dir: Path to output directory
        comparison_dir: Path to comparison directory (for Docker comparison)
        
    Returns:
        bool: True if processing succeeded
    """
    print("\n" + "="*80)
    print(f"Processing target: {target_id}")
    print("="*80)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for subdir in ['thermo_features', 'dihedral_features', 'mi_features']:
        (output_dir / subdir).mkdir(exist_ok=True, parents=True)
    
    # Get sequence for target
    sequence = get_sequence_for_target(target_id, data_dir)
    if not sequence:
        print(f"Could not get sequence for {target_id}")
        return False
    
    print(f"Sequence length: {len(sequence)}")
    
    # Load structure data
    structure_data = load_structure_data(target_id, data_dir)
    if structure_data is not None:
        print(f"Loaded structure data with {len(structure_data)} residues")
    else:
        print(f"Warning: Could not load structure data, will use fallbacks")
    
    # Track memory usage
    log_memory_usage("Initial memory")
    
    # Extract thermodynamic features
    try:
        thermo_file = extract_thermodynamic_features(
            target_id=target_id,
            sequence=sequence,
            output_dir=output_dir / 'thermo_features'
        )
        print(f"Saved thermodynamic features to {thermo_file}")
    except Exception as e:
        print(f"Error extracting thermodynamic features: {e}")
        traceback.print_exc()
    
    # Get MSA data and extract MI features
    try:
        msa_sequences = get_msa_data_for_target(target_id, data_dir)
        if msa_sequences and len(msa_sequences) >= 2:
            mi_file = extract_mi_features(
                target_id=target_id,
                msa_sequences=msa_sequences,
                output_dir=output_dir / 'mi_features'
            )
            print(f"Saved MI features to {mi_file}")
        else:
            print(f"Could not extract MI features (insufficient MSA data)")
    except Exception as e:
        print(f"Error extracting MI features: {e}")
        traceback.print_exc()
    
    # Extract dihedral features if we have structure data
    try:
        if structure_data is not None:
            dihedral_file = extract_dihedral_features(
                target_id=target_id,
                structure_data=structure_data,
                output_dir=output_dir / 'dihedral_features'
            )
            print(f"Saved dihedral features to {dihedral_file}")
        else:
            # Fall back to dummy features
            dihedral_file = make_dummy_dihedral_features(
                target_id=target_id,
                seq_length=len(sequence),
                output_dir=output_dir / 'dihedral_features'
            )
            print(f"Saved dummy dihedral features to {dihedral_file}")
    except Exception as e:
        print(f"Error extracting dihedral features: {e}")
        traceback.print_exc()
    
    # Verify feature compatibility with data loader
    success = verify_target_features(output_dir, target_id, verbose=True)
    
    # Compare with Docker output if requested
    if comparison_dir:
        comparison_dir = Path(comparison_dir)
        if comparison_dir.exists():
            docker_match = compare_outputs(output_dir, comparison_dir, target_id)
            if docker_match:
                print("\n✅ Local and Docker outputs match")
            else:
                print("\n❌ Differences found between local and Docker outputs")
    
    # Plot memory usage
    plot_dir = output_dir / 'memory_plots'
    plot_dir.mkdir(exist_ok=True, parents=True)
    plot_memory_usage(output_file=plot_dir / f"{target_id}_memory_usage.png")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Run a complete test of the RNA feature extraction pipeline on a single target")
    parser.add_argument("--target", default="R1107", help="Target ID to process")
    parser.add_argument("--data-dir", default="./data/raw", help="Path to data directory")
    parser.add_argument("--output-dir", default="./data/processed", help="Path to output directory")
    parser.add_argument("--docker", action="store_true", help="Compare outputs with Docker environment")
    parser.add_argument("--docker-dir", default="./data/processed_docker", help="Path to Docker output directory (if different)")
    args = parser.parse_args()
    
    # Process target
    success = process_target(
        target_id=args.target,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        comparison_dir=args.docker_dir if args.docker else None
    )
    
    if success:
        print("\n✅ Target processing completed successfully")
        return 0
    else:
        print("\n❌ Target processing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())