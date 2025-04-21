"""
DataManager Module

This module handles data loading, saving, and format conversion for RNA feature extraction.
It provides functionality to load RNA sequences from CSV files, MSA data from FASTA files,
and save/load extracted features.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import logging

class DataManager:
    """
    Handles data loading, saving, and format conversion for RNA feature extraction.
    """
    
    def __init__(self, data_dir=None, raw_dir=None, processed_dir=None):
        """
        Initialize with configurable data directories.
        
        Args:
            data_dir (Path, optional): Base data directory. Defaults to None.
            raw_dir (Path, optional): Directory for raw data. Defaults to None.
            processed_dir (Path, optional): Directory for processed features. Defaults to None.
        """
        # Set default paths if not provided
        if data_dir is None:
            self.data_dir = Path("data")
        else:
            self.data_dir = Path(data_dir)
            
        if raw_dir is None:
            self.raw_dir = self.data_dir / "raw"
        else:
            self.raw_dir = Path(raw_dir)
            
        if processed_dir is None:
            self.processed_dir = self.data_dir / "processed"
        else:
            self.processed_dir = Path(processed_dir)
            
        # Create output directories if they don't exist
        self.thermo_dir = self.processed_dir / "thermo_features"
        self.mi_dir = self.processed_dir / "mi_features"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Setup logging
        self.logger = logging.getLogger("DataManager")
        
    def _ensure_directories(self):
        """
        Ensure that all necessary directories exist.
        """
        for directory in [self.data_dir, self.raw_dir, self.processed_dir, 
                          self.thermo_dir, self.mi_dir]:
            directory.mkdir(exist_ok=True, parents=True)
            
    def load_rna_data(self, csv_path):
        """
        Load RNA data from CSV file.
        
        Args:
            csv_path (str or Path): Path to CSV file containing RNA data
            
        Returns:
            DataFrame: DataFrame with RNA data or None if loading failed
        """
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded {len(df)} entries from {csv_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            return None
            
    def get_unique_target_ids(self, df, id_col="ID"):
        """
        Extract unique target IDs from dataframe.
        
        Args:
            df (DataFrame): DataFrame with RNA data
            id_col (str, optional): Column containing IDs. Defaults to "ID".
            
        Returns:
            list: List of unique target IDs
        """
        # Extract target IDs (format: TARGET_ID_RESIDUE_NUM)
        target_ids = []
        for id_str in df[id_col]:
            # Split the ID string and get the target ID part
            id_str = str(id_str)
            parts = id_str.split('_')
            
            if len(parts) >= 2:
                # Format: R1107_1 -> extract R1107 as the target ID
                if parts[0].startswith('R') and parts[1].isdigit():
                    # Special case for R-style IDs with numeric second part
                    target_id = parts[0]
                else:
                    # Standard case: Take the first two parts (e.g., "1SCL_A")
                    target_id = f"{parts[0]}_{parts[1]}"
            else:
                # Single part ID or unusual format - use as is
                target_id = id_str
                
            target_ids.append(target_id)
        
        # Get unique target IDs
        unique_targets = sorted(list(set(target_ids)))
        self.logger.info(f"Found {len(unique_targets)} unique target IDs")
        return unique_targets
        
    def load_msa_data(self, target_id, data_dir=None):
        """
        Load MSA data for a given target.
        
        Args:
            target_id (str): Target ID
            data_dir (Path, optional): Directory containing MSA data. Defaults to None.
            
        Returns:
            list: List of MSA sequences or None if not found
        """
        if data_dir is None:
            data_dir = self.raw_dir
            
        # Define possible MSA directories and extensions
        msa_dirs = [
            data_dir / "MSA",
            data_dir,
            data_dir / "alignments",
            data_dir / "test" / "MSA",
            data_dir / "test",
            data_dir / "test" / "alignments"
        ]
        
        extensions = [".MSA.fasta", ".fasta", ".fa", ".afa", ".msa"]
        
        # Try all combinations of directories and extensions
        for msa_dir in msa_dirs:
            if not msa_dir.exists():
                continue
                
            for ext in extensions:
                msa_path = msa_dir / f"{target_id}{ext}"
                if msa_path.exists():
                    self.logger.info(f"Loading MSA data from {msa_path}")
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
                        
                        self.logger.info(f"Loaded {len(sequences)} sequences from MSA")
                        return sequences
                    except Exception as e:
                        self.logger.error(f"Error loading MSA data: {e}")
        
        # Fallback: try recursive search
        self.logger.info(f"MSA file not found in standard locations, trying recursive search...")
        try:
            for msa_dir in [data_dir, data_dir / "test"]:
                if not msa_dir.exists():
                    continue
                    
                for ext in extensions:
                    pattern = f"**/{target_id}{ext}"
                    matches = list(msa_dir.glob(pattern))
                    if matches:
                        msa_path = matches[0]
                        self.logger.info(f"Found MSA via recursive search: {msa_path}")
                        
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
                        
                        self.logger.info(f"Loaded {len(sequences)} sequences from MSA")
                        return sequences
        except Exception as e:
            self.logger.error(f"Error in recursive MSA search: {e}")
        
        self.logger.warning(f"Could not find MSA data for {target_id}")
        return None
        
    def get_sequence_for_target(self, target_id, data_dir=None):
        """
        Get RNA sequence for a target ID from the sequence file.
        
        Args:
            target_id (str): Target ID
            data_dir (Path, optional): Directory containing sequence data. Defaults to None.
            
        Returns:
            str: RNA sequence as string or None if not found
        """
        if data_dir is None:
            data_dir = self.raw_dir
            
        # Try different possible file locations
        sequence_paths = [
            data_dir / "test" / "sequences.csv",
            data_dir / "test" / "test_sequences.csv",
            data_dir / "test" / "rna_sequences.csv",
            data_dir / "test_sequences.csv",
            data_dir / "sequences.csv"
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
                                        return sequence
                except Exception as e:
                    self.logger.error(f"Error loading sequence data from {path}: {e}")
        
        # If we still haven't found the sequence, try to extract it from MSA data
        msa_sequences = self.load_msa_data(target_id, data_dir)
        if msa_sequences and len(msa_sequences) > 0:
            # The first sequence in the MSA is typically the target sequence
            return msa_sequences[0]
        
        self.logger.warning(f"Could not find sequence for {target_id}")
        return None
        
    def save_features(self, features, output_file):
        """
        Save extracted features to NPZ file.
        
        Args:
            features (dict): Dictionary of features to save
            output_file (str or Path): Path to save features
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            np.savez_compressed(output_file, **features)
            self.logger.info(f"Saved features to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving features: {e}")
            return False
            
    def load_features(self, target_id, feature_type=None):
        """
        Load features for a target ID.
        
        Args:
            target_id (str): Target ID
            feature_type (str, optional): Type of features to load ('thermo', 'mi').
                                         If None, load all types. Defaults to None.
            
        Returns:
            dict: Dictionary with loaded features or None if loading failed
        """
        if feature_type is None:
            # Load all feature types
            features = {}
            
            # Load thermodynamic features
            thermo_file = self.thermo_dir / f"{target_id}_thermo_features.npz"
            if thermo_file.exists():
                try:
                    features['thermo'] = dict(np.load(thermo_file, allow_pickle=True))
                    self.logger.info(f"Loaded thermodynamic features for {target_id}")
                except Exception as e:
                    self.logger.error(f"Error loading thermodynamic features for {target_id}: {e}")
            
            # Load MI features
            mi_file = self.mi_dir / f"{target_id}_mi_features.npz"
            if mi_file.exists():
                try:
                    features['mi'] = dict(np.load(mi_file, allow_pickle=True))
                    self.logger.info(f"Loaded MI features for {target_id}")
                except Exception as e:
                    self.logger.error(f"Error loading MI features for {target_id}: {e}")
            
            return features if features else None
            
        else:
            # Load specific feature type
            try:
                # Determine file path based on feature type
                if feature_type == 'thermo':
                    file_path = self.thermo_dir / f"{target_id}_thermo_features.npz"
                elif feature_type == 'mi':
                    file_path = self.mi_dir / f"{target_id}_mi_features.npz"
                else:
                    self.logger.error(f"Unknown feature type: {feature_type}")
                    return None
                    
                # Check if file exists
                if not file_path.exists():
                    self.logger.warning(f"Feature file not found: {file_path}")
                    return None
                    
                # Load features
                features = dict(np.load(file_path, allow_pickle=True))
                self.logger.info(f"Loaded {feature_type} features for {target_id}")
                return features
                
            except Exception as e:
                self.logger.error(f"Error loading features for {target_id}: {e}")
                return None