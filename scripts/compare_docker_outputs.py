#!/usr/bin/env python3
"""
Docker Output Comparison Script

This script compares feature outputs between local and Docker environments to ensure
consistency. It checks both file existence and content similarity.

Usage:
    python compare_docker_outputs.py [--local-dir LOCAL_DIR] [--docker-dir DOCKER_DIR] [--target TARGET_ID]

Author: Claude
Date: April 12, 2025
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json
import traceback

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

def find_common_targets(local_dir, docker_dir):
    """Find target IDs that exist in both directories."""
    local_dir = Path(local_dir)
    docker_dir = Path(docker_dir)
    
    local_targets = set(find_target_ids(local_dir))
    docker_targets = set(find_target_ids(docker_dir))
    
    common_targets = local_targets.intersection(docker_targets)
    return sorted(list(common_targets))

def compare_feature_files(local_file, docker_file):
    """Compare two feature files for content similarity."""
    results = {
        'exists_in_both': False,
        'keys_match': False,
        'shapes_match': False,
        'values_match': False,
        'differences': []
    }
    
    # Check existence
    if not local_file.exists():
        results['differences'].append(f"Local file missing: {local_file}")
        return results
    
    if not docker_file.exists():
        results['differences'].append(f"Docker file missing: {docker_file}")
        return results
    
    results['exists_in_both'] = True
    
    try:
        # Load data from both files
        local_data = dict(np.load(local_file, allow_pickle=True))
        docker_data = dict(np.load(docker_file, allow_pickle=True))
        
        # Compare keys
        local_keys = set(local_data.keys())
        docker_keys = set(docker_data.keys())
        
        if local_keys != docker_keys:
            missing_in_docker = local_keys - docker_keys
            missing_in_local = docker_keys - local_keys
            
            if missing_in_docker:
                results['differences'].append(f"Keys missing in Docker: {', '.join(missing_in_docker)}")
            
            if missing_in_local:
                results['differences'].append(f"Keys missing in local: {', '.join(missing_in_local)}")
        else:
            results['keys_match'] = True
        
        # Compare shapes for array features
        shape_mismatches = []
        
        for key in local_keys.intersection(docker_keys):
            if isinstance(local_data[key], np.ndarray) and isinstance(docker_data[key], np.ndarray):
                if local_data[key].shape != docker_data[key].shape:
                    shape_mismatches.append(
                        f"{key}: local {local_data[key].shape} vs docker {docker_data[key].shape}"
                    )
        
        if shape_mismatches:
            results['differences'].append(f"Shape mismatches: {'; '.join(shape_mismatches)}")
        else:
            results['shapes_match'] = True
        
        # Compare key numeric values
        value_mismatches = []
        
        for key in ['mfe', 'ensemble_energy', 'raw_ensemble_energy', 'mean_entropy']:
            if key in local_data and key in docker_data:
                local_val = float(local_data[key])
                docker_val = float(docker_data[key])
                
                # Allow small differences due to numerical precision
                if abs(local_val - docker_val) > 0.01:
                    value_mismatches.append(f"{key}: local {local_val} vs docker {docker_val}")
        
        if value_mismatches:
            results['differences'].append(f"Value mismatches: {'; '.join(value_mismatches)}")
        else:
            results['values_match'] = True
        
        return results
        
    except Exception as e:
        results['differences'].append(f"Error comparing files: {str(e)}")
        return results

def compare_target_outputs(target_id, local_dir, docker_dir):
    """Compare all feature files for a specific target."""
    local_dir = Path(local_dir)
    docker_dir = Path(docker_dir)
    
    print(f"\nComparing outputs for target: {target_id}")
    
    # Define feature types and expected file names
    feature_types = {
        'thermo': f"{target_id}_thermo_features.npz",
        'dihedral': f"{target_id}_dihedral_features.npz",
        'mi': f"{target_id}_features.npz"
    }
    
    # Map feature types to subdirectories
    subdirs = {
        'thermo': 'thermo_features',
        'dihedral': 'dihedral_features',
        'mi': 'mi_features'
    }
    
    # Compare each feature type
    results = {}
    all_match = True
    
    for feature_type, file_name in feature_types.items():
        subdir = subdirs[feature_type]
        local_file = local_dir / subdir / file_name
        docker_file = docker_dir / subdir / file_name
        
        # Compare the files
        comparison = compare_feature_files(local_file, docker_file)
        results[feature_type] = comparison
        
        # Print results
        if comparison['exists_in_both'] and comparison['keys_match'] and comparison['shapes_match'] and comparison['values_match']:
            print(f"✅ {feature_type.title()}: Files match")
        else:
            print(f"❌ {feature_type.title()}: Differences found")
            for diff in comparison['differences']:
                print(f"   - {diff}")
            all_match = False
    
    return all_match, results

def run_full_comparison(local_dir, docker_dir, target_id=None):
    """Run a comparison of all feature outputs between local and Docker environments."""
    local_dir = Path(local_dir)
    docker_dir = Path(docker_dir)
    
    print("\n" + "="*80)
    print(f"Comparing features between local and Docker environments")
    print(f"Local directory: {local_dir}")
    print(f"Docker directory: {docker_dir}")
    print("="*80)
    
    # Find targets to compare
    if target_id:
        targets = [target_id]
    else:
        targets = find_common_targets(local_dir, docker_dir)
    
    if not targets:
        print("\nNo common targets found between local and Docker directories.")
        return False
    
    print(f"\nFound {len(targets)} targets to compare")
    
    # Compare each target
    match_count = 0
    results = {}
    
    for target_id in targets:
        all_match, target_results = compare_target_outputs(target_id, local_dir, docker_dir)
        results[target_id] = target_results
        
        if all_match:
            match_count += 1
    
    # Print summary
    print("\n" + "="*80)
    print(f"Comparison Summary:")
    print(f"- Total targets: {len(targets)}")
    print(f"- Matching targets: {match_count}")
    print(f"- Targets with differences: {len(targets) - match_count}")
    
    if match_count == len(targets):
        print("\n✅ All outputs match between local and Docker environments")
        return True
    else:
        print(f"\n⚠️ {len(targets) - match_count} targets have differences between environments")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compare feature outputs between local and Docker environments")
    parser.add_argument("--local-dir", default="../data/processed", help="Path to local processed features directory")
    parser.add_argument("--docker-dir", default="../data/processed_docker", help="Path to Docker processed features directory")
    parser.add_argument("--target", help="Compare a specific target ID (optional)")
    args = parser.parse_args()
    
    # Convert to Path objects
    local_dir = Path(args.local_dir)
    docker_dir = Path(args.docker_dir)
    
    # Check if directories exist
    if not local_dir.exists() or not local_dir.is_dir():
        print(f"Error: Local directory '{local_dir}' does not exist or is not a directory")
        return 1
    
    if not docker_dir.exists() or not docker_dir.is_dir():
        print(f"Error: Docker directory '{docker_dir}' does not exist or is not a directory")
        return 1
    
    # Run comparison
    success = run_full_comparison(local_dir, docker_dir, args.target)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())