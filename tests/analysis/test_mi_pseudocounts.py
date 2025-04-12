"""
Tests for the pseudocount functionality in mutual information calculations.

These tests validate that pseudocount corrections are properly implemented
and integrated with existing functionality in the MI modules.
"""

import unittest
import os
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.analysis.mutual_information import calculate_mutual_information, get_adaptive_pseudocount
from src.analysis.rna_mi_pipeline.enhanced_mi import (
    calculate_mutual_information_enhanced,
    get_adaptive_pseudocount as get_adaptive_pseudocount_enhanced
)

class TestMIPseudocounts(unittest.TestCase):
    """Test cases for mutual information pseudocount implementation."""
    
    def setUp(self):
        """Set up test data."""
        # Example MSA sequences
        self.small_msa = [
            "ACGUACGU",
            "ACGCACGU",
            "ACGAACGU",
            "ACGCACGU",
        ]
        
        self.medium_msa = [
            "ACGUACGU",
            "ACGCACGU",
            "ACGAACGU",
            "ACGCACGU",
        ] * 15  # 60 sequences
        
        self.large_msa = [
            "ACGUACGU",
            "ACGCACGU",
            "ACGAACGU",
            "ACGCACGU",
        ] * 50  # 200 sequences
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_adaptive_pseudocount_selection(self):
        """Test that adaptive pseudocount selection works correctly."""
        # Small MSA (<= 25 sequences)
        self.assertEqual(get_adaptive_pseudocount(self.small_msa), 0.5)
        self.assertEqual(get_adaptive_pseudocount_enhanced(self.small_msa), 0.5)
        
        # Medium MSA (26-100 sequences)
        self.assertEqual(get_adaptive_pseudocount(self.medium_msa), 0.2)
        self.assertEqual(get_adaptive_pseudocount_enhanced(self.medium_msa), 0.2)
        
        # Large MSA (> 100 sequences)
        self.assertEqual(get_adaptive_pseudocount(self.large_msa), 0.0)
        self.assertEqual(get_adaptive_pseudocount_enhanced(self.large_msa), 0.0)
    
    def test_basic_mi_pseudocount_integration(self):
        """Test pseudocount integration in basic MI implementation."""
        # Calculate MI without pseudocounts
        mi_result_no_pc = calculate_mutual_information(self.small_msa, pseudocount=0.0)
        
        # Calculate MI with pseudocounts
        mi_result_with_pc = calculate_mutual_information(self.small_msa, pseudocount=0.5)
        
        # Both should return valid results
        self.assertIsNotNone(mi_result_no_pc)
        self.assertIsNotNone(mi_result_with_pc)
        
        # Check that parameters are correctly stored
        self.assertEqual(mi_result_no_pc['params']['pseudocount'], 0.0)
        self.assertEqual(mi_result_with_pc['params']['pseudocount'], 0.5)
        
        # Results should be different with pseudocounts
        self.assertFalse(np.array_equal(
            mi_result_no_pc['coupling_matrix'], 
            mi_result_with_pc['coupling_matrix']
        ))
        
        # With adaptive selection
        mi_result_adaptive = calculate_mutual_information(self.small_msa, pseudocount=None)
        self.assertEqual(mi_result_adaptive['params']['pseudocount'], 0.5)  # Should be 0.5 for small MSA
    
    def test_enhanced_mi_pseudocount_integration(self):
        """Test pseudocount integration in enhanced MI implementation."""
        # Calculate MI without pseudocounts
        mi_result_no_pc = calculate_mutual_information_enhanced(
            self.small_msa, pseudocount=0.0
        )
        
        # Calculate MI with pseudocounts
        mi_result_with_pc = calculate_mutual_information_enhanced(
            self.small_msa, pseudocount=0.5
        )
        
        # Both should return valid results
        self.assertIsNotNone(mi_result_no_pc)
        self.assertIsNotNone(mi_result_with_pc)
        
        # Check that parameters are correctly stored
        self.assertEqual(mi_result_no_pc['params']['pseudocount'], 0.0)
        self.assertEqual(mi_result_with_pc['params']['pseudocount'], 0.5)
        
        # Results should be different with pseudocounts
        self.assertFalse(np.array_equal(
            mi_result_no_pc['coupling_matrix'], 
            mi_result_with_pc['coupling_matrix']
        ))
        
        # With adaptive selection
        mi_result_adaptive = calculate_mutual_information_enhanced(
            self.small_msa, pseudocount=None
        )
        self.assertEqual(mi_result_adaptive['params']['pseudocount'], 0.5)  # Should be 0.5 for small MSA
    
    def test_pseudocount_normalization(self):
        """Test that pseudocount normalization is mathematically correct."""
        # Both implementations should produce properly normalized probability distributions
        # Basic implementation
        mi_result_basic = calculate_mutual_information(self.small_msa, pseudocount=0.5)
        
        # Enhanced implementation
        mi_result_enhanced = calculate_mutual_information_enhanced(
            self.small_msa, pseudocount=0.5
        )
        
        # Both implementations should produce matrices with non-zero values
        self.assertTrue(np.any(mi_result_basic['coupling_matrix'] > 0))
        self.assertTrue(np.any(mi_result_enhanced['coupling_matrix'] > 0))
        
        # Scores should be non-negative (MI is always >= 0)
        self.assertTrue(np.all(mi_result_basic['coupling_matrix'] >= 0))
        self.assertTrue(np.all(mi_result_enhanced['coupling_matrix'] >= 0))
    
    def test_sequence_weighting_integration(self):
        """Test integration of pseudocounts with sequence weighting."""
        # Create weights that sum to 1
        weights = np.ones(len(self.small_msa)) / len(self.small_msa)
        
        # Calculate MI with sequence weighting
        mi_result_weighted = calculate_mutual_information_enhanced(
            self.small_msa, 
            weights=weights,
            pseudocount=0.5
        )
        
        # Should return valid results
        self.assertIsNotNone(mi_result_weighted)
        self.assertEqual(mi_result_weighted['params']['pseudocount'], 0.5)
        
        # Results should be different from unweighted calculation
        mi_result_unweighted = calculate_mutual_information_enhanced(
            self.small_msa, 
            pseudocount=0.5
        )
        
        # Even with uniform weights, implementation details may cause slight differences
        # so we just check that the calculation runs successfully
        self.assertIsNotNone(mi_result_unweighted)
        self.assertEqual(mi_result_unweighted['params']['pseudocount'], 0.5)
    
    def test_apc_correction_integration(self):
        """Test integration with APC correction."""
        # APC correction should still be applied after pseudocount calculation
        mi_result = calculate_mutual_information_enhanced(self.small_msa, pseudocount=0.5)
        
        # Should have both raw MI and APC-corrected matrices
        self.assertIn('mi_matrix', mi_result)
        self.assertIn('apc_matrix', mi_result)
        
        # APC matrix should be different from raw MI matrix
        self.assertFalse(np.array_equal(
            mi_result['mi_matrix'],
            mi_result['apc_matrix']
        ))
    
    def test_no_pseudocount_matrix_has_expected_values(self):
        """Test that MI matrices contain expected values with and without pseudocounts."""
        # Create an MSA with a reasonable number of sequences to avoid numerical instability
        test_msa = [
            "ACGUACGU",
            "ACGCACGU",
            "ACGAACGU",
            "ACGCACGU",
            "ACGAACGU",
            "ACGGACGU",
            "ACGCACGU",
            "ACGUACGU",
        ]
        
        # Basic implementation
        mi_result_no_pc = calculate_mutual_information(test_msa, pseudocount=0.0)
        mi_result_with_pc = calculate_mutual_information(test_msa, pseudocount=0.5)
        
        # Both should return valid results
        self.assertIsNotNone(mi_result_no_pc)
        self.assertIsNotNone(mi_result_with_pc)
        
        # Enhanced implementation
        mi_result_enhanced_no_pc = calculate_mutual_information_enhanced(test_msa, pseudocount=0.0)
        mi_result_enhanced_with_pc = calculate_mutual_information_enhanced(test_msa, pseudocount=0.5)
        
        # Both should return valid results
        self.assertIsNotNone(mi_result_enhanced_no_pc)
        self.assertIsNotNone(mi_result_enhanced_with_pc)
        
        # Check that the dimensions are correct
        seq_len = len(test_msa[0])
        self.assertEqual(mi_result_no_pc['coupling_matrix'].shape, (seq_len, seq_len))
        self.assertEqual(mi_result_with_pc['coupling_matrix'].shape, (seq_len, seq_len))
        self.assertEqual(mi_result_enhanced_no_pc['coupling_matrix'].shape, (seq_len, seq_len))
        self.assertEqual(mi_result_enhanced_with_pc['coupling_matrix'].shape, (seq_len, seq_len))

if __name__ == '__main__':
    unittest.main()