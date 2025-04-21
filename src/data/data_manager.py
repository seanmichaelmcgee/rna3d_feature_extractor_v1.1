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
        # Placeholder implementation
        return []
        
    def load_msa_data(self, target_id, data_dir=None):
        """
        Load MSA data for a given target.
        
        Args:
            target_id (str): Target ID
            data_dir (Path, optional): Directory containing MSA data. Defaults to None.
            
        Returns:
            list: List of MSA sequences or None if not found
        """
        # Placeholder implementation
        return []
        
    def get_sequence_for_target(self, target_id, data_dir=None):
        """
        Get RNA sequence for a target ID from the sequence file.
        
        Args:
            target_id (str): Target ID
            data_dir (Path, optional): Directory containing sequence data. Defaults to None.
            
        Returns:
            str: RNA sequence as string or None if not found
        """
        # Placeholder implementation
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
            
    def load_features(self, target_id, feature_type):
        """
        Load features for a target ID.
        
        Args:
            target_id (str): Target ID
            feature_type (str): Type of features to load ('thermo', 'mi')
            
        Returns:
            dict: Dictionary with loaded features or None if loading failed
        """
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