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
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

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
        
        # Create test CSV file for sequences
        self.test_csv = self.raw_dir / "test_sequences.csv"
        self.test_df = pd.DataFrame({
            'ID': ['R1107', 'R1108', 'R1116'],
            'sequence': ['ACGUGCGUGA', 'UGCGUGCAAU', 'AUUGUGCAAUUGCAUGCAUAU']
        })
        self.test_df.to_csv(self.test_csv, index=False)
        
        # Create test MSA directory and file
        self.msa_dir = self.raw_dir / "MSA"
        self.msa_dir.mkdir(exist_ok=True)
        
        self.msa_content = (
            ">seq1\n"
            "ACGUGCGUGA\n"
            ">seq2\n"
            "ACGCGCGUGA\n"
            ">seq3\n"
            "ACGCACGUGA\n"
        )
        
        self.msa_file = self.msa_dir / "R1107.MSA.fasta"
        with open(self.msa_file, 'w') as f:
            f.write(self.msa_content)
        
        # Create test features
        self.test_thermo_features = {
            'target_id': 'R1107',
            'mfe': -10.5,
            'ensemble_energy': -11.2,
            'sequence': 'ACGUGCGUGA'
        }
        
        self.test_mi_features = {
            'target_id': 'R1107',
            'scores': np.zeros((10, 10)),
            'coupling_matrix': np.zeros((10, 10)),
            'method': 'mutual_information'
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
        
        # Check paths are correct
        self.assertEqual(self.data_manager.data_dir, Path(self.test_dir))
        self.assertEqual(self.data_manager.raw_dir, Path(self.raw_dir))
        self.assertEqual(self.data_manager.processed_dir, Path(self.processed_dir))
        
    def test_load_rna_data(self):
        """Test loading RNA data from CSV."""
        # Load test CSV
        df = self.data_manager.load_rna_data(self.test_csv)
        
        # Check DataFrame
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertIn('ID', df.columns)
        self.assertIn('sequence', df.columns)
        
        # Check values
        self.assertEqual(df['ID'].iloc[0], 'R1107')
        self.assertEqual(df['sequence'].iloc[0], 'ACGUGCGUGA')
        
    def test_get_unique_target_ids(self):
        """Test extracting unique target IDs from dataframe."""
        # Create test dataframe with complex IDs
        complex_df = pd.DataFrame({
            'ID': ['R1107_1', 'R1107_2', 'R1108_1', 'R1116_1']
        })
        
        # Get unique target IDs
        target_ids = self.data_manager.get_unique_target_ids(complex_df)
        
        # Check result
        self.assertEqual(len(target_ids), 3)  # R1107, R1108, R1116
        self.assertIn('R1107', target_ids)
        self.assertIn('R1108', target_ids)
        self.assertIn('R1116', target_ids)
        
        # Test with different format IDs
        mixed_df = pd.DataFrame({
            'ID': ['R1107_1', '1SCL_A_1', '1SCL_A_2']
        })
        
        mixed_target_ids = self.data_manager.get_unique_target_ids(mixed_df)
        self.assertEqual(len(mixed_target_ids), 2)  # R1107, 1SCL_A
        self.assertIn('R1107', mixed_target_ids)
        self.assertIn('1SCL_A', mixed_target_ids)
        
    def test_load_msa_data(self):
        """Test loading MSA data."""
        # Load MSA data
        sequences = self.data_manager.load_msa_data('R1107')
        
        # Check result
        self.assertIsNotNone(sequences)
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0], 'ACGUGCGUGA')
        self.assertEqual(sequences[1], 'ACGCGCGUGA')
        self.assertEqual(sequences[2], 'ACGCACGUGA')
        
    def test_get_sequence_for_target(self):
        """Test getting sequence for target."""
        # Get sequence from CSV
        sequence = self.data_manager.get_sequence_for_target('R1107')
        
        # Check result
        self.assertIsNotNone(sequence)
        self.assertEqual(sequence, 'ACGUGCGUGA')
        
        # Test fallback to MSA for unknown target in CSV
        # Create a new MSA file for a target not in the CSV
        msa_content = (
            ">seq1\n"
            "GCUAGCUAGCUA\n"
            ">seq2\n"
            "GCUAGCUAGCUA\n"
        )
        
        new_msa_file = self.msa_dir / "R1200.MSA.fasta"
        with open(new_msa_file, 'w') as f:
            f.write(msa_content)
            
        # Get sequence for target not in CSV
        sequence = self.data_manager.get_sequence_for_target('R1200')
        
        # Check result
        self.assertIsNotNone(sequence)
        self.assertEqual(sequence, 'GCUAGCUAGCUA')
        
    def test_save_and_load_features(self):
        """Test saving and loading features."""
        # Save thermo features
        thermo_file = self.data_manager.thermo_dir / "R1107_thermo_features.npz"
        result = self.data_manager.save_features(self.test_thermo_features, thermo_file)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(thermo_file.exists())
        
        # Save MI features
        mi_file = self.data_manager.mi_dir / "R1107_mi_features.npz"
        result = self.data_manager.save_features(self.test_mi_features, mi_file)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(mi_file.exists())
        
        # Load specific feature type
        thermo_features = self.data_manager.load_features('R1107', 'thermo')
        
        # Check loaded features
        self.assertIsNotNone(thermo_features)
        self.assertEqual(thermo_features['target_id'], 'R1107')
        self.assertEqual(thermo_features['mfe'], -10.5)
        self.assertEqual(thermo_features['ensemble_energy'], -11.2)
        self.assertEqual(thermo_features['sequence'], 'ACGUGCGUGA')
        
        # Load MI features
        mi_features = self.data_manager.load_features('R1107', 'mi')
        
        # Check loaded features
        self.assertIsNotNone(mi_features)
        self.assertEqual(mi_features['target_id'], 'R1107')
        self.assertEqual(mi_features['method'], 'mutual_information')
        self.assertTrue('scores' in mi_features)
        self.assertTrue('coupling_matrix' in mi_features)
        
        # Load all features
        all_features = self.data_manager.load_features('R1107')
        
        # Check all features
        self.assertIsNotNone(all_features)
        self.assertTrue('thermo' in all_features)
        self.assertTrue('mi' in all_features)
        self.assertEqual(all_features['thermo']['target_id'], 'R1107')
        self.assertEqual(all_features['mi']['target_id'], 'R1107')


if __name__ == '__main__':
    unittest.main()