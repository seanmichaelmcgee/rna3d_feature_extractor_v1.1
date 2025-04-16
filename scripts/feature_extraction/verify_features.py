#!/usr/bin/env python3
"""
Feature verification script for testing the feature extraction scripts.
This script verifies that the extracted features have the correct format
and contain the expected data.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

def verify_thermo_features(feature_file):
    """Verify thermodynamic features"""
    try:
        data = np.load(feature_file, allow_pickle=True)
        print(f"Thermodynamic features for {Path(feature_file).stem}:")
        
        # Expected keys
        expected_keys = ['target_id', 'sequence', 'mfe', 'ensemble_energy', 'pairing_probs']
        missing_keys = [key for key in expected_keys if key not in data]
        
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys}")
            return False
        
        # Basic checks
        print(f"  - Target ID: {data['target_id']}")
        print(f"  - Sequence length: {len(data['sequence'])}")
        print(f"  - MFE: {data['mfe']:.2f}")
        print(f"  - Ensemble energy: {data['ensemble_energy']:.2f}")
        
        # Check pairing probabilities
        if 'pairing_probs' in data:
            probs = data['pairing_probs']
            seq_len = len(data['sequence'])
            if probs.shape != (seq_len, seq_len):
                print(f"⚠️ Pairing probability matrix has incorrect shape: {probs.shape}, expected ({seq_len}, {seq_len})")
                return False
            print(f"  - Pairing probability matrix: {probs.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error verifying thermodynamic features: {e}")
        return False

def verify_dihedral_features(feature_file):
    """Verify dihedral features"""
    try:
        data = np.load(feature_file, allow_pickle=True)
        print(f"Dihedral features for {Path(feature_file).stem}:")
        
        # Check for multi-structure format or single structure
        if 'num_structures' in data:
            print(f"  - Multiple structure format detected")
            print(f"  - Number of structures: {data['num_structures']}")
            
            # Check structure IDs
            if 'structure_ids' in data:
                print(f"  - Structure IDs: {data['structure_ids']}")
                
            # Check if at least one structure has features
            found_features = False
            for key in data.keys():
                if key.startswith('struct_') and key.endswith('_features'):
                    found_features = True
                    features = data[key]
                    print(f"  - Structure features shape: {features.shape}")
                    break
                    
            if not found_features:
                print("⚠️ No structure features found")
                return False
        else:
            # Basic single structure checks
            if 'features' not in data:
                print("⚠️ Missing 'features' key")
                return False
                
            features = data['features']
            print(f"  - Features shape: {features.shape}")
            
            # Verify shape - should be (n, 4) for dihedral angles
            if len(features.shape) != 2 or features.shape[1] != 4:
                print(f"⚠️ Features have incorrect shape: {features.shape}, expected (n, 4)")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error verifying dihedral features: {e}")
        return False

def verify_mi_features(feature_file):
    """Verify mutual information features"""
    try:
        data = np.load(feature_file, allow_pickle=True)
        print(f"MI features for {Path(feature_file).stem}:")
        
        # Expected keys in MI features
        has_scores = 'scores' in data
        has_coupling = 'coupling_matrix' in data
        
        if not (has_scores or has_coupling):
            print("⚠️ Missing both 'scores' and 'coupling_matrix' keys")
            return False
        
        # Check the MI matrix
        if has_scores:
            matrix = data['scores']
            print(f"  - MI scores matrix shape: {matrix.shape}")
            # MI matrix should be square
            if matrix.shape[0] != matrix.shape[1]:
                print(f"⚠️ MI matrix is not square: {matrix.shape}")
                return False
        elif has_coupling:
            matrix = data['coupling_matrix']
            print(f"  - Coupling matrix shape: {matrix.shape}")
            if matrix.shape[0] != matrix.shape[1]:
                print(f"⚠️ Coupling matrix is not square: {matrix.shape}")
                return False
        
        # Check if top pairs exist
        if 'top_pairs' in data:
            top_pairs = data['top_pairs']
            print(f"  - Top pairs available: {len(top_pairs)} pairs")
        
        return True
    except Exception as e:
        print(f"❌ Error verifying MI features: {e}")
        return False

def verify_directory(directory):
    """Verify all feature files in a directory"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory {directory} does not exist")
        return False
    
    print(f"\nVerifying features in {directory}")
    
    # Verify thermodynamic features
    thermo_dir = directory / "thermo_features"
    if thermo_dir.exists():
        print(f"\n=== Thermodynamic Features ===")
        thermo_files = list(thermo_dir.glob("*_thermo_features.npz"))
        if not thermo_files:
            print("No thermodynamic feature files found")
        else:
            print(f"Found {len(thermo_files)} thermodynamic feature files")
            for file in thermo_files[:2]:  # Check first 2 files for brevity
                verify_thermo_features(file)
    
    # Verify dihedral features
    dihedral_dir = directory / "dihedral_features"
    if dihedral_dir.exists():
        print(f"\n=== Dihedral Features ===")
        dihedral_files = list(dihedral_dir.glob("*_dihedral_features.npz"))
        if not dihedral_files:
            print("No dihedral feature files found")
        else:
            print(f"Found {len(dihedral_files)} dihedral feature files")
            for file in dihedral_files[:2]:  # Check first 2 files for brevity
                verify_dihedral_features(file)
    
    # Verify MI features
    mi_dir = directory / "mi_features"
    if mi_dir.exists():
        print(f"\n=== MI Features ===")
        mi_files = list(mi_dir.glob("*_mi_features.npz"))
        if not mi_files:
            print("No MI feature files found")
        else:
            print(f"Found {len(mi_files)} MI feature files")
            for file in mi_files[:2]:  # Check first 2 files for brevity
                verify_mi_features(file)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify extracted RNA feature files")
    parser.add_argument("directory", help="Directory containing feature files")
    parser.add_argument("--thermo-file", help="Specific thermodynamic feature file to verify")
    parser.add_argument("--dihedral-file", help="Specific dihedral feature file to verify")
    parser.add_argument("--mi-file", help="Specific MI feature file to verify")
    
    args = parser.parse_args()
    
    # Verify specific files if provided
    if args.thermo_file:
        print(f"Verifying thermodynamic feature file: {args.thermo_file}")
        verify_thermo_features(args.thermo_file)
    
    if args.dihedral_file:
        print(f"Verifying dihedral feature file: {args.dihedral_file}")
        verify_dihedral_features(args.dihedral_file)
    
    if args.mi_file:
        print(f"Verifying MI feature file: {args.mi_file}")
        verify_mi_features(args.mi_file)
    
    # Verify entire directory if no specific files provided
    if not (args.thermo_file or args.dihedral_file or args.mi_file):
        verify_directory(args.directory)
    
    print("\nVerification complete")

if __name__ == "__main__":
    main()