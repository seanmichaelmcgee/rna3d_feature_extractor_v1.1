#!/usr/bin/env python3
"""
Feature Verification Script

This script checks if the extracted RNA features are compatible with the PyTorch data loader
requirements by verifying:
1. Correct directory structure
2. Proper file naming conventions
3. Expected feature names and array shapes
4. Data type compatibility

Usage:
    python verify_feature_compatibility.py /path/to/processed/features

Author: Claude
Date: April 12, 2025
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
import argparse
import traceback

def verify_directory_structure(features_dir):
    """Verify the required directory structure exists."""
    features_dir = Path(features_dir)
    required_dirs = ['dihedral_features', 'thermo_features', 'mi_features']
    
    print(f"\nChecking directory structure in {features_dir}...")
    missing_dirs = []
    
    for d in required_dirs:
        dir_path = features_dir / d
        if not dir_path.exists() or not dir_path.is_dir():
            missing_dirs.append(d)
            print(f"❌ Missing required directory: {d}")
        else:
            print(f"✅ Found required directory: {d}")
    
    if missing_dirs:
        print(f"\n⚠️ WARNING: Missing {len(missing_dirs)} required directories")
        print("The data loader expects the following structure:")
        print("features_dir/")
        print("├── dihedral_features/")
        print("├── thermo_features/")
        print("└── mi_features/")
        return False
    
    return True

def find_target_ids(features_dir):
    """Find all target IDs with at least one feature file."""
    features_dir = Path(features_dir)
    target_ids = set()
    
    # Check each feature directory
    for feature_type in ['dihedral_features', 'thermo_features', 'mi_features']:
        dir_path = features_dir / feature_type
        if not dir_path.exists():
            continue
        
        # Find all NPZ files and extract target IDs
        for file_path in dir_path.glob('*.npz'):
            file_name = file_path.name
            
            # Extract target ID based on expected naming pattern
            if feature_type == 'dihedral_features' and file_name.endswith('_dihedral_features.npz'):
                target_id = file_name[:-len('_dihedral_features.npz')]
                target_ids.add(target_id)
            elif feature_type == 'thermo_features' and file_name.endswith('_thermo_features.npz'):
                target_id = file_name[:-len('_thermo_features.npz')]
                target_ids.add(target_id)
            elif feature_type == 'mi_features' and file_name.endswith('_features.npz'):
                target_id = file_name[:-len('_features.npz')]
                target_ids.add(target_id)
                
    return sorted(list(target_ids))

def verify_file_naming(features_dir, target_id):
    """Verify file naming follows the required pattern for a target ID."""
    features_dir = Path(features_dir)
    expected_files = {
        'dihedral': features_dir / 'dihedral_features' / f"{target_id}_dihedral_features.npz",
        'thermo': features_dir / 'thermo_features' / f"{target_id}_thermo_features.npz",
        'mi': features_dir / 'mi_features' / f"{target_id}_features.npz"
    }
    
    found_files = {}
    for feature_type, file_path in expected_files.items():
        if file_path.exists():
            found_files[feature_type] = file_path
    
    return found_files

def verify_feature_shape(file_path, feature_type):
    """Verify feature arrays have the expected shapes and names."""
    try:
        # Load the NPZ file
        data = np.load(file_path, allow_pickle=True)
        
        # Check based on feature type
        if feature_type == 'dihedral':
            # Dihedral features should be (sequence_length, 4)
            if 'features' not in data:
                return False, "Missing 'features' key in dihedral features"
            
            features = data['features']
            if len(features.shape) != 2 or features.shape[1] != 4:
                return False, f"Dihedral features have incorrect shape: {features.shape}, expected (sequence_length, 4)"
            
            return True, f"Dihedral features valid with shape {features.shape}"
            
        elif feature_type == 'thermo':
            # Check for pairing probabilities (base_pair_probs or pairing_probs)
            if 'pairing_probs' in data:
                pairing_probs = data['pairing_probs']
            elif 'base_pair_probs' in data:
                pairing_probs = data['base_pair_probs']
            else:
                return False, "Missing pairing probability matrix (need 'pairing_probs' or 'base_pair_probs')"
            
            # Pairing probabilities should be (sequence_length, sequence_length)
            if len(pairing_probs.shape) != 2 or pairing_probs.shape[0] != pairing_probs.shape[1]:
                return False, f"Pairing probability matrix has incorrect shape: {pairing_probs.shape}, expected square matrix"
            
            # Check positional entropy
            if 'positional_entropy' in data:
                pos_entropy = data['positional_entropy']
            elif 'position_entropy' in data:
                pos_entropy = data['position_entropy']
            else:
                return False, "Missing positional entropy (need 'positional_entropy' or 'position_entropy')"
            
            # Positional entropy should be (sequence_length,)
            if len(pos_entropy.shape) != 1 or pos_entropy.shape[0] != pairing_probs.shape[0]:
                return False, f"Positional entropy has incorrect shape: {pos_entropy.shape}, expected ({pairing_probs.shape[0]},)"
            
            return True, f"Thermodynamic features valid with matrix shape {pairing_probs.shape}"
            
        elif feature_type == 'mi':
            # Check for coupling matrix (coupling_matrix or scores)
            if 'coupling_matrix' in data:
                coupling_matrix = data['coupling_matrix']
            elif 'scores' in data:
                coupling_matrix = data['scores']
            else:
                return False, "Missing coupling matrix (need 'coupling_matrix' or 'scores')"
            
            # Coupling matrix should be (sequence_length, sequence_length)
            if len(coupling_matrix.shape) != 2 or coupling_matrix.shape[0] != coupling_matrix.shape[1]:
                return False, f"Coupling matrix has incorrect shape: {coupling_matrix.shape}, expected square matrix"
            
            return True, f"MI features valid with matrix shape {coupling_matrix.shape}"
            
        return False, f"Unknown feature type: {feature_type}"
        
    except Exception as e:
        return False, f"Error checking {file_path}: {str(e)}"

def load_features_with_data_loader(features_dir, target_id):
    """Simulate loading features with the data loader to verify compatibility."""
    try:
        features_dir = Path(features_dir)
        result = {}
        
        # Load dihedral features
        dihedral_path = features_dir / 'dihedral_features' / f"{target_id}_dihedral_features.npz"
        if dihedral_path.exists():
            dihedral_data = dict(np.load(dihedral_path, allow_pickle=True))
            if 'features' in dihedral_data:
                result['dihedral'] = {
                    'features': dihedral_data['features']
                }
        
        # Load thermodynamic features
        thermo_path = features_dir / 'thermo_features' / f"{target_id}_thermo_features.npz"
        if thermo_path.exists():
            thermo_data = dict(np.load(thermo_path, allow_pickle=True))
            result['thermo'] = {}
            
            # Standardize feature names for compatibility
            if 'pairing_probs' in thermo_data:
                result['thermo']['pairing_probs'] = thermo_data['pairing_probs']
            elif 'base_pair_probs' in thermo_data:
                result['thermo']['pairing_probs'] = thermo_data['base_pair_probs']
                
            if 'positional_entropy' in thermo_data:
                result['thermo']['positional_entropy'] = thermo_data['positional_entropy']
            elif 'position_entropy' in thermo_data:
                result['thermo']['positional_entropy'] = thermo_data['position_entropy']
                
            # Include accessibility if available
            if 'accessibility' in thermo_data:
                result['thermo']['accessibility'] = thermo_data['accessibility']
        
        # Load MI features
        mi_path = features_dir / 'mi_features' / f"{target_id}_features.npz"
        if mi_path.exists():
            mi_data = dict(np.load(mi_path, allow_pickle=True))
            result['evolutionary'] = {}
            
            # Standardize feature names for compatibility
            if 'coupling_matrix' in mi_data:
                result['evolutionary']['coupling_matrix'] = mi_data['coupling_matrix']
            elif 'scores' in mi_data:
                result['evolutionary']['coupling_matrix'] = mi_data['scores']
        
        # Check if we loaded any features
        if not result:
            return False, "No features loaded"
            
        # Verify the expected keys exist
        expected_groups = ['dihedral', 'thermo', 'evolutionary']
        missing_groups = [g for g in expected_groups if g not in result]
        
        if missing_groups:
            return len(missing_groups) < len(expected_groups), f"Missing feature groups: {', '.join(missing_groups)}"
            
        return True, result
        
    except Exception as e:
        traceback.print_exc()
        return False, f"Error loading features: {str(e)}"

def verify_target_features(features_dir, target_id, verbose=False):
    """Verify all features for a single target ID."""
    print(f"\nVerifying features for target ID: {target_id}")
    
    # Check file naming
    found_files = verify_file_naming(features_dir, target_id)
    if not found_files:
        print(f"❌ No feature files found for {target_id}")
        return False
    
    print(f"Found {len(found_files)}/3 feature types:")
    
    # Verify feature shapes
    all_valid = True
    for feature_type, file_path in found_files.items():
        valid, message = verify_feature_shape(file_path, feature_type)
        if valid:
            print(f"✅ {feature_type.title()} features: {message}")
        else:
            print(f"❌ {feature_type.title()} features: {message}")
            all_valid = False
    
    # Simulate data loader
    loader_valid, loader_result = load_features_with_data_loader(features_dir, target_id)
    if loader_valid:
        print(f"✅ Data loader simulation successful")
        if verbose and isinstance(loader_result, dict):
            print("  Loaded feature groups:")
            for group, features in loader_result.items():
                print(f"  - {group}: {', '.join(features.keys())}")
    else:
        print(f"❌ Data loader simulation failed: {loader_result}")
        all_valid = False
    
    return all_valid

def verify_all_features(features_dir, verbose=False):
    """Verify features for all targets found in the directory."""
    print("\n" + "="*80)
    print(f"Verifying RNA Feature Compatibility in: {features_dir}")
    print("="*80)
    
    # Verify directory structure
    if not verify_directory_structure(features_dir):
        return False
    
    # Find all target IDs
    target_ids = find_target_ids(features_dir)
    if not target_ids:
        print("\n⚠️ No target IDs found in the features directory")
        return False
    
    print(f"\nFound {len(target_ids)} targets with features")
    if verbose:
        print(f"Target IDs: {', '.join(target_ids[:5])}" + ("..." if len(target_ids) > 5 else ""))
    
    # Verify features for each target
    valid_targets = 0
    for target_id in target_ids:
        if verify_target_features(features_dir, target_id, verbose):
            valid_targets += 1
    
    # Print summary
    print("\n" + "="*80)
    print(f"Feature Verification Summary:")
    print(f"- Total targets: {len(target_ids)}")
    print(f"- Valid targets: {valid_targets}")
    print(f"- Invalid targets: {len(target_ids) - valid_targets}")
    
    if valid_targets == len(target_ids):
        print("\n✅ All features are compatible with the data loader")
        return True
    elif valid_targets > 0:
        print(f"\n⚠️ {valid_targets}/{len(target_ids)} targets have compatible features")
        return True
    else:
        print("\n❌ No targets have fully compatible features")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify RNA feature compatibility with data loader")
    parser.add_argument("features_dir", help="Path to the processed features directory")
    parser.add_argument("--target", help="Verify a specific target ID (optional)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    args = parser.parse_args()
    
    # Convert to Path object
    features_dir = Path(args.features_dir)
    if not features_dir.exists() or not features_dir.is_dir():
        print(f"Error: Features directory '{features_dir}' does not exist or is not a directory")
        return 1
    
    # Verify features
    if args.target:
        success = verify_target_features(features_dir, args.target, args.verbose)
    else:
        success = verify_all_features(features_dir, args.verbose)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())