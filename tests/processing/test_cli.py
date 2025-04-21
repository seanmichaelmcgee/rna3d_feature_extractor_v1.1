"""
Tests for the batch processing CLI.
"""

import unittest
import os
import tempfile
import shutil
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

from src.processing.cli import main, load_targets_from_file, load_targets_from_csv

class TestBatchProcessingCLI(unittest.TestCase):
    """Tests for the batch processing CLI."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "processed")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a test targets file
        self.targets_file = os.path.join(self.temp_dir, "targets.txt")
        with open(self.targets_file, 'w') as f:
            f.write("target1\ntarget2\ntarget3\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_targets_from_file(self):
        """Test loading targets from a file."""
        targets = load_targets_from_file(self.targets_file)
        self.assertEqual(targets, ["target1", "target2", "target3"])
    
    @patch('pandas.read_csv')
    def test_load_targets_from_csv(self, mock_read_csv):
        """Test loading targets from a CSV file."""
        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["ID", "other_column"]
        mock_df.__getitem__.return_value.unique.return_value = ["target1", "target2"]
        mock_read_csv.return_value = mock_df
        
        targets = load_targets_from_csv("dummy.csv")
        self.assertEqual(targets, ["target1", "target2"])
    
    @patch('src.processing.cli.BatchProcessor')
    @patch('sys.argv', ['cli.py', '--target', 'target1', '--extract-thermo', '--extract-mi'])
    def test_main_single_target(self, mock_batch_processor):
        """Test main function with a single target."""
        # Mock BatchProcessor instance
        mock_processor = MagicMock()
        mock_processor.process_targets.return_value = {
            "batch_name": "test_batch",
            "total_targets": 1,
            "successful_targets": 1,
            "skipped_targets": 0,
        }
        mock_batch_processor.return_value = mock_processor
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            main()
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__
        
        # Check that process_targets was called with the right arguments
        mock_processor.process_targets.assert_called_once()
        call_args = mock_processor.process_targets.call_args[1]
        self.assertEqual(call_args["target_ids"], ["target1"])
        self.assertTrue(call_args["extract_thermo"])
        self.assertTrue(call_args["extract_mi"])
    
    @patch('src.processing.cli.BatchProcessor')
    @patch('sys.argv', ['cli.py', '--targets-file', 'dummy_path', '--extract-thermo'])
    def test_main_targets_file(self, mock_batch_processor):
        """Test main function with targets from a file."""
        # Mock BatchProcessor instance
        mock_processor = MagicMock()
        mock_processor.process_targets.return_value = {
            "batch_name": "test_batch",
            "total_targets": 3,
            "successful_targets": 3,
            "skipped_targets": 0,
        }
        mock_batch_processor.return_value = mock_processor
        
        # Mock load_targets_from_file
        with patch('src.processing.cli.load_targets_from_file') as mock_load:
            mock_load.return_value = ["target1", "target2", "target3"]
            
            # Capture stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                main()
            finally:
                # Reset stdout
                sys.stdout = sys.__stdout__
            
            # Check that process_targets was called with the right arguments
            mock_processor.process_targets.assert_called_once()
            call_args = mock_processor.process_targets.call_args[1]
            self.assertEqual(call_args["target_ids"], ["target1", "target2", "target3"])
            self.assertTrue(call_args["extract_thermo"])
            self.assertFalse(call_args["extract_mi"])

if __name__ == '__main__':
    unittest.main()