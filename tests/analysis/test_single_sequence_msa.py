"""
Tests for single-sequence MSA optimization in mutual information calculation.
"""

import unittest
import numpy as np
import time
from src.analysis.mutual_information import calculate_mutual_information
from src.analysis.rna_mi_pipeline.enhanced_mi import calculate_mutual_information_enhanced, chunk_and_analyze_rna

class TestSingleSequenceMSA(unittest.TestCase):
    """Test case for the single-sequence MSA optimization."""

    def test_single_sequence_detection_basic_mi(self):
        """Test that single-sequence MSA is properly detected in basic MI calculation."""
        # Create a single-sequence MSA
        single_seq_msa = ["ACGUCGAUCGAUCGA"]
        
        # Calculate MI with the single sequence
        result = calculate_mutual_information(single_seq_msa)
        
        # Check if the output has the expected structure
        self.assertIsNotNone(result)
        self.assertIn('scores', result)
        self.assertIn('coupling_matrix', result)
        self.assertIn('method', result)
        self.assertIn('params', result)
        
        # Check if the single_sequence flag was set correctly
        self.assertIn('single_sequence', result['params'])
        self.assertTrue(result['params']['single_sequence'])
        
        # Check if the matrix has all zeros
        self.assertEqual(np.sum(result['scores']), 0.0)
        self.assertEqual(np.sum(result['coupling_matrix']), 0.0)
        
        # Check if top_pairs is empty
        self.assertEqual(len(result['top_pairs']), 0)

    def test_identical_sequences_basic_mi(self):
        """Test that multiple identical sequences are handled correctly in basic MI."""
        # Create an MSA with multiple identical sequences
        identical_seq_msa = ["ACGUCGAUCGAUCGA", "ACGUCGAUCGAUCGA", "ACGUCGAUCGAUCGA"]
        
        # Calculate MI with identical sequences
        result = calculate_mutual_information(identical_seq_msa)
        
        # Check if the single_sequence flag was set correctly
        self.assertIn('single_sequence', result['params'])
        self.assertTrue(result['params']['single_sequence'])
        
        # Check if the matrix has all zeros
        self.assertEqual(np.sum(result['scores']), 0.0)

    def test_performance_improvement_basic_mi(self):
        """Test that the optimization improves performance for long sequences."""
        # Create a long single-sequence MSA
        long_seq = "A" * 3000
        long_single_seq_msa = [long_seq]
        
        # Measure time with optimization
        start_time = time.time()
        result = calculate_mutual_information(long_single_seq_msa)
        optimized_time = time.time() - start_time
        
        # The optimized version should be very fast
        self.assertLess(optimized_time, 0.5, "Single-sequence optimization should be fast")
        
        # Check if the output is as expected
        self.assertEqual(result['scores'].shape, (3000, 3000))
        self.assertEqual(np.sum(result['scores']), 0.0)

    def test_single_sequence_detection_enhanced_mi(self):
        """Test that single-sequence MSA is properly detected in enhanced MI calculation."""
        # Create a single-sequence MSA
        single_seq_msa = ["ACGUCGAUCGAUCGA"]
        
        # Calculate enhanced MI with the single sequence
        result = calculate_mutual_information_enhanced(single_seq_msa)
        
        # Check if the output has the expected structure
        self.assertIsNotNone(result)
        self.assertIn('mi_matrix', result)
        self.assertIn('apc_matrix', result)
        self.assertIn('scores', result)
        self.assertIn('coupling_matrix', result)
        self.assertIn('method', result)
        self.assertIn('params', result)
        
        # Check if the single_sequence flag was set correctly
        self.assertIn('single_sequence', result['params'])
        self.assertTrue(result['params']['single_sequence'])
        
        # Check if the matrices have all zeros
        self.assertEqual(np.sum(result['mi_matrix']), 0.0)
        self.assertEqual(np.sum(result['apc_matrix']), 0.0)
        self.assertEqual(np.sum(result['scores']), 0.0)
        self.assertEqual(np.sum(result['coupling_matrix']), 0.0)
        
        # Check if top_pairs is empty
        self.assertEqual(len(result['top_pairs']), 0)
        
    def test_chunking_with_single_sequence(self):
        """Test that chunking correctly handles single-sequence MSAs."""
        # Create a long single-sequence MSA
        long_seq = "A" * 1000
        long_single_seq_msa = [long_seq]
        
        # Run with chunking
        result = chunk_and_analyze_rna(
            long_single_seq_msa,
            max_length=500,  # Force chunking by setting max_length less than sequence length
            verbose=True
        )
        
        # Check if the output has the expected structure
        self.assertIsNotNone(result)
        self.assertIn('params', result)
        self.assertIn('single_sequence', result['params'])
        self.assertTrue(result['params']['single_sequence'])
        
        # Check if the matrices have all zeros
        self.assertEqual(np.sum(result['scores']), 0.0)
        
        # The optimization should have bypassed the chunking process
        self.assertNotIn('chunks', result)
        
    def test_normal_msa_unaffected(self):
        """Test that normal MSAs with multiple different sequences are unaffected."""
        # Create a normal MSA with different sequences
        normal_msa = [
            "ACGUCGAUCGAUCGA",
            "ACGUCGAUCGAUCCA",  # Different from first sequence
            "ACGUCGAUCGAUCAA"   # Different from both
        ]
        
        # Calculate MI with the normal MSA
        result_basic = calculate_mutual_information(normal_msa)
        result_enhanced = calculate_mutual_information_enhanced(normal_msa)
        
        # Check that the single_sequence flag is not present or false
        self.assertNotIn('single_sequence', result_basic['params'])
        self.assertNotIn('single_sequence', result_enhanced['params'])
        
        # Check that the matrices are not all zeros
        self.assertGreater(np.sum(result_basic['scores']), 0.0)
        self.assertGreater(np.sum(result_enhanced['scores']), 0.0)

if __name__ == '__main__':
    unittest.main()