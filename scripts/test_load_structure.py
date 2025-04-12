#!/usr/bin/env python3
"""
Test script to verify structure data loading from labels CSV files.

Run with: python scripts/test_load_structure.py TARGET_ID
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_structure_data(target_id, data_dir=None):
    """
    Load structure data for a given target from labels CSV.
    
    Args:
        target_id: Target ID
        data_dir: Directory containing data (optional)
        
    Returns:
        DataFrame with structure coordinates or None if not found
    """
    if data_dir is None:
        # Default to project's data directory
        data_dir = Path("data/raw")
    else:
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

def test_load_structure(target_id):
    """Test loading structure data for a specific target ID."""
    # Attempt to load structure data directly from label files
    label_files = [
        Path("data/raw/train_labels.csv"),
        Path("data/raw/validation_labels.csv")
    ]
    
    for label_file in label_files:
        if not label_file.exists():
            print(f"Warning: {label_file} does not exist")
            continue
            
        try:
            all_data = pd.read_csv(label_file)
            target_data = all_data[all_data["ID"].str.startswith(f"{target_id}_")]
            
            if len(target_data) > 0:
                print(f"✅ Found {len(target_data)} residues for {target_id} in {label_file}")
                print("First few rows:")
                print(target_data.head())
                
                # Check for required coordinate columns
                coord_cols = [col for col in target_data.columns if col.startswith(('x_', 'y_', 'z_'))]
                if coord_cols:
                    print(f"✅ Found coordinate columns: {coord_cols[:6]}...")
                else:
                    print("❌ No coordinate columns found!")
                
                # Now test the load_structure_data function
                print("\nTesting load_structure_data function:")
                loaded_data = load_structure_data(target_id)
                if loaded_data is not None:
                    print(f"✅ Function successfully loaded {len(loaded_data)} residues")
                    
                    # Compare shapes to ensure we got the same data
                    if len(loaded_data) == len(target_data):
                        print(f"✅ Row counts match: {len(loaded_data)}")
                    else:
                        print(f"❌ Row counts don't match: {len(loaded_data)} vs {len(target_data)}")
                else:
                    print("❌ Function failed to load structure data")
                
                return
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    print(f"❌ Could not find data for {target_id} in any labels file")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_load_structure.py TARGET_ID")
        sys.exit(1)
        
    target_id = sys.argv[1]
    test_load_structure(target_id)