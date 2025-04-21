"""
Tests for the DataManager class.
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from src.data.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    """Test cases for DataManager."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create subdirectories
        self.raw_dir = Path(self.test_dir) / "raw"
        self.processed_dir = Path(self.test_dir) / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Create test CSV file
        self.test_csv = self.raw_dir / "test_sequences.csv"
        df = pd.DataFrame({
            'ID': ['R1107', 'R1108', 'R1116'],
            'sequence': ['ACGUGCGUGA', 'UGCGUGCAAU', 'AUUGUGCAAUUGCAUGCAUAU']
        })
        df.to_csv(self.test_csv, index=False)
        
        # Create test features
        self.test_features = {
            'target_id': 'R1107',
            'mfe': -10.5,
            'ensemble_energy': -11.2,
            'sequence': 'ACGUGCGUGA'
        }
        
        # Initialize DataManager
        self.data_manager = DataManager(
            data_dir=self.test_dir,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir
        )
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test DataManager initialization and directory creation."""
        # Check directories exist
        self.assertTrue(self.data_manager.data_dir.exists())
        self.assertTrue(self.data_manager.raw_dir.exists())
        self.assertTrue(self.data_manager.processed_dir.exists())
        self.assertTrue(self.data_manager.thermo_dir.exists())
        self.assertTrue(self.data_manager.mi_dir.exists())
        
    def test_load_rna_data(self):
        """Test loading RNA data from CSV."""
        # Load test CSV
        df = self.data_manager.load_rna_data(self.test_csv)
        
        # Check DataFrame
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertIn('ID', df.columns)
        self.assertIn('sequence', df.columns)
        
    def test_save_and_load_features(self):
        """Test saving and loading features."""
        # Save features
        output_file = self.data_manager.thermo_dir / "R1107_thermo_features.npz"
        result = self.data_manager.save_features(self.test_features, output_file)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(output_file.exists())
        
        # Load features
        loaded_features = self.data_manager.load_features('R1107', 'thermo')
        
        # Check loaded features
        self.assertIsNotNone(loaded_features)
        self.assertEqual(loaded_features['target_id'], 'R1107')
        self.assertEqual(loaded_features['mfe'], -10.5)
        self.assertEqual(loaded_features['ensemble_energy'], -11.2)
        self.assertEqual(loaded_features['sequence'], 'ACGUGCGUGA')


if __name__ == '__main__':
    unittest.main()