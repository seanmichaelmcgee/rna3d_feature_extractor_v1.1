"""
Integration tests for the RNA Feature Extraction Workflow.

These tests verify that all components of the system work together correctly.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch

from src.data.data_manager import DataManager
from src.features.feature_extractor import FeatureExtractor
from src.processing.batch_processor import BatchProcessor
from src.analysis.memory_monitor import MemoryMonitor
from src.validation.result_validator import ResultValidator
from src.workflow import RNAFeatureExtractionWorkflow

class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for the RNA feature extraction workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, "data")
        self.test_output_dir = os.path.join(self.temp_dir, "processed")
        self.test_log_dir = os.path.join(self.temp_dir, "logs")
        
        # Create directories
        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_log_dir, exist_ok=True)
        
        # Create a test targets file
        self.targets_file = os.path.join(self.test_data_dir, "test_targets.txt")
        with open(self.targets_file, 'w') as f:
            f.write("test_target_1\ntest_target_2\ntest_target_3\n")
        
        # Create mock components
        self.mock_data_manager = MagicMock(spec=DataManager)
        self.mock_feature_extractor = MagicMock(spec=FeatureExtractor)
        self.mock_memory_monitor = MagicMock(spec=MemoryMonitor)
        self.mock_result_validator = MagicMock(spec=ResultValidator)
        
        # Configure mock data manager
        self.mock_data_manager.get_sequence_for_target.return_value = "GAUCGAUCGAUC"
        self.mock_data_manager.load_msa_data.return_value = ["GAUCGAUCGAUC", "GAUCGAUCGAUC"]
        
        # Configure mock feature extractor
        thermo_features = {
            "struct.mfe": np.array(-10.5),
            "struct.ensemble_energy": np.array(-10.0),
            "struct.pairing_probs": np.eye(12),
        }
        
        mi_features = {
            "mi.scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "mi.coupling_matrix": np.ones((12, 12)) * 0.1,
        }
        
        self.mock_feature_extractor.extract_thermodynamic_features.return_value = thermo_features
        self.mock_feature_extractor.extract_mi_features.return_value = mi_features
        self.mock_feature_extractor.extract_features.return_value = {
            "thermo_features": thermo_features,
            "mi_features": mi_features,
        }
        
        # Configure mock memory monitor
        self.mock_memory_monitor.get_current_memory_usage.return_value = 2.0
        self.mock_memory_monitor.start_monitoring.return_value = 2.0
        self.mock_memory_monitor.stop_monitoring.return_value = 3.0
        self.mock_memory_monitor.is_monitoring.return_value = True
        
        # Configure mock result validator
        self.mock_result_validator.validate_thermodynamic_features.return_value = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "stats": {"mfe": -10.5, "ensemble_energy": -10.0},
        }
        
        self.mock_result_validator.validate_mi_features.return_value = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "stats": {"max_mi": 0.1, "mean_mi": 0.1},
        }
        
        self.mock_result_validator.validate_target_features.return_value = {
            "target_id": "test_target_1",
            "is_valid": True,
            "feature_results": {},
            "issues": [],
            "warnings": [],
        }
        
        self.mock_result_validator.validate_batch_results.return_value = {
            "batch_name": "test_batch",
            "total_targets": 3,
            "valid_targets": 3,
            "invalid_targets": 0,
            "targets_with_warnings": 0,
            "target_results": {},
            "summary": {"issues_by_type": {}, "warnings_by_type": {}},
        }
        
        # Create workflow with mock components
        self.workflow = RNAFeatureExtractionWorkflow(
            data_dir=self.test_data_dir,
            output_dir=self.test_output_dir,
            log_dir=self.test_log_dir,
            memory_plot_dir=os.path.join(self.test_output_dir, "memory_plots"),
            validation_report_dir=os.path.join(self.test_output_dir, "validation_reports"),
            max_memory_gb=16.0,
            batch_size=10,
        )
        
        # Replace components with mocks
        self.workflow.data_manager = self.mock_data_manager
        self.workflow.feature_extractor = self.mock_feature_extractor
        self.workflow.memory_monitor = self.mock_memory_monitor
        self.workflow.result_validator = self.mock_result_validator
        
        # Create a new batch processor with mocks
        self.workflow.batch_processor = BatchProcessor(
            data_manager=self.mock_data_manager,
            feature_extractor=self.mock_feature_extractor,
            memory_monitor=self.mock_memory_monitor,
            output_dir=self.test_output_dir,
            log_dir=self.test_log_dir,
            max_memory_usage_gb=16.0,
            batch_size=10,
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_workflow_initialization(self):
        """Test initialization of the workflow."""
        self.assertEqual(self.workflow.data_dir, self.test_data_dir)
        self.assertEqual(self.workflow.output_dir, self.test_output_dir)
        self.assertEqual(self.workflow.log_dir, self.test_log_dir)
        self.assertEqual(self.workflow.max_memory_gb, 16.0)
        self.assertEqual(self.workflow.batch_size, 10)
    
    def test_single_target_extraction(self):
        """Test extraction of features for a single target."""
        target_id = "test_target_1"
        
        results = self.workflow.extract_single_target(
            target_id=target_id,
            extract_thermo=True,
            extract_mi=True,
            validate_results=True,
            save_memory_plot=True,
        )
        
        # Check that all components were called correctly
        self.mock_data_manager.get_sequence_for_target.assert_called_with(target_id)
        self.mock_data_manager.load_msa_data.assert_called_with(target_id)
        self.mock_feature_extractor.extract_features.assert_called_with(
            target_id=target_id,
            sequence=self.mock_data_manager.get_sequence_for_target.return_value,
            msa_sequences=self.mock_data_manager.load_msa_data.return_value,
            extract_thermo=True,
            extract_mi=True,
        )
        self.mock_memory_monitor.start_monitoring.assert_called()
        self.mock_memory_monitor.stop_monitoring.assert_called()
        self.mock_memory_monitor.plot_memory_usage.assert_called()
        self.mock_result_validator.validate_target_features.assert_called_with(
            target_id=target_id,
            data_dir=self.test_output_dir,
        )
        
        # Check that data_manager.save_features was called for each feature type
        self.mock_data_manager.save_features.assert_called()
        
        # Check that the results contain the expected keys
        self.assertEqual(results["target_id"], target_id)
        self.assertIn("execution_time", results)
        self.assertIn("peak_memory_gb", results)
        self.assertIn("features_extracted", results)
        self.assertIn("validation", results)
    
    @patch('builtins.open')
    def test_batch_extraction(self, mock_open):
        """Test batch extraction of features."""
        # Configure the mock_open to return our targets file content
        mock_open.return_value.__enter__.return_value.readlines.return_value = [
            "test_target_1\n", "test_target_2\n", "test_target_3\n"
        ]
        
        # Create a mock for the batch processor process_targets method
        self.workflow.batch_processor.process_targets = MagicMock(return_value={
            "batch_name": "test_batch",
            "total_targets": 3,
            "successful_targets": 3,
            "skipped_targets": 0,
            "skipped_target_ids": [],
            "results": {
                "test_target_1": {"status": "success"},
                "test_target_2": {"status": "success"},
                "test_target_3": {"status": "success"},
            },
        })
        
        # Run the workflow
        results = self.workflow.run_extraction(
            targets_file=self.targets_file,
            extract_thermo=True,
            extract_mi=True,
            batch_name="test_batch",
            validate_results=True,
            resume=False,
            save_memory_plots=True,
        )
        
        # Check that all components were called correctly
        self.workflow.batch_processor.process_targets.assert_called_with(
            target_ids=["test_target_1", "test_target_2", "test_target_3"],
            extract_thermo=True,
            extract_mi=True,
            save_intermediates=True,
            batch_name="test_batch",
        )
        self.mock_memory_monitor.start_monitoring.assert_called()
        self.mock_memory_monitor.stop_monitoring.assert_called()
        self.mock_memory_monitor.plot_memory_usage.assert_called()
        self.mock_result_validator.validate_batch_results.assert_called()
        
        # Check that the results contain the expected keys
        self.assertEqual(results["batch_name"], "test_batch")
        self.assertEqual(results["total_targets"], 3)
        self.assertEqual(results["successful_targets"], 3)
        self.assertEqual(results["skipped_targets"], 0)
        self.assertIn("execution_time", results)
        self.assertIn("peak_memory_gb", results)
        self.assertIn("validation", results)
    
    @patch('builtins.open')
    def test_resume_batch_extraction(self, mock_open):
        """Test resuming a batch extraction."""
        # Configure the mock_open to return our targets file content
        mock_open.return_value.__enter__.return_value.readlines.return_value = [
            "test_target_1\n", "test_target_2\n", "test_target_3\n"
        ]
        
        # Create a mock for the batch processor resume_batch_processing method
        self.workflow.batch_processor.resume_batch_processing = MagicMock(return_value={
            "batch_name": "test_batch",
            "total_targets": 3,
            "successful_targets": 3,
            "skipped_targets": 0,
            "skipped_target_ids": [],
            "results": {
                "test_target_1": {"status": "success"},
                "test_target_2": {"status": "success"},
                "test_target_3": {"status": "success"},
            },
        })
        
        # Run the workflow with resume=True
        results = self.workflow.run_extraction(
            targets_file=self.targets_file,
            extract_thermo=True,
            extract_mi=True,
            batch_name="test_batch",
            validate_results=True,
            resume=True,
            save_memory_plots=True,
        )
        
        # Check that all components were called correctly
        self.workflow.batch_processor.resume_batch_processing.assert_called_with(
            batch_name="test_batch",
            extract_thermo=True,
            extract_mi=True,
            save_intermediates=True,
        )
        self.mock_memory_monitor.start_monitoring.assert_called()
        self.mock_memory_monitor.stop_monitoring.assert_called()
        self.mock_memory_monitor.plot_memory_usage.assert_called()
        self.mock_result_validator.validate_batch_results.assert_called()
        
        # Check that the results contain the expected keys
        self.assertEqual(results["batch_name"], "test_batch")
        self.assertEqual(results["total_targets"], 3)
        self.assertEqual(results["successful_targets"], 3)
        self.assertEqual(results["skipped_targets"], 0)
        self.assertIn("execution_time", results)
        self.assertIn("peak_memory_gb", results)
        self.assertIn("validation", results)

if __name__ == "__main__":
    unittest.main()