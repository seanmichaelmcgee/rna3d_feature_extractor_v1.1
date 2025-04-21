"""
Tests for the ResultValidator class.
"""

import unittest
import os
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import MagicMock, patch

from src.validation.result_validator import ResultValidator
from src.data.data_manager import DataManager

class TestResultValidator(unittest.TestCase):
    """Tests for the ResultValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create mock DataManager
        self.mock_data_manager = MagicMock(spec=DataManager)
        
        # Configure mock to return test data
        def mock_load_features(target_id, feature_type=None, data_dir=None):
            if feature_type == "thermo":
                return {
                    "struct.mfe": np.array(-10.5),
                    "struct.ensemble_energy": np.array(-10.0),
                    "struct.pairing_probs": np.eye(10),  # 10x10 identity matrix
                }
            elif feature_type == "mi":
                return {
                    "mi.scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                    "mi.coupling_matrix": np.ones((10, 10)) * 0.1,  # 10x10 matrix of 0.1s
                }
            elif feature_type == "dihedral":
                return {
                    "geom.dihedrals": np.random.rand(10, 4) * 2 * np.pi - np.pi,  # 10x4 random dihedrals
                }
            return {}
        
        self.mock_data_manager.load_features.side_effect = mock_load_features
        
        # Create ResultValidator with mock DataManager
        self.validator = ResultValidator(data_manager=self.mock_data_manager)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of ResultValidator."""
        self.assertIsNotNone(self.validator)
        self.assertEqual(self.validator.data_manager, self.mock_data_manager)
        self.assertTrue("entropy_threshold" in self.validator.config)
    
    def test_validate_thermodynamic_features(self):
        """Test validation of thermodynamic features."""
        # Create valid thermodynamic features
        valid_features = {
            "struct.mfe": np.array(-10.5),
            "struct.ensemble_energy": np.array(-10.0),
            "struct.pairing_probs": np.eye(10),  # 10x10 identity matrix
        }
        
        results = self.validator.validate_thermodynamic_features(valid_features)
        self.assertTrue(results["is_valid"])
        self.assertEqual(len(results["issues"]), 0)
        
        # Create invalid thermodynamic features (MFE > ensemble energy)
        invalid_features = {
            "struct.mfe": np.array(-9.5),  # MFE higher than ensemble energy
            "struct.ensemble_energy": np.array(-10.0),
            "struct.pairing_probs": np.eye(10),
        }
        
        results = self.validator.validate_thermodynamic_features(invalid_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("Thermodynamic inconsistency" in issue for issue in results["issues"]))
        
        # Create features with NaN values
        nan_features = {
            "struct.mfe": np.array(np.nan),
            "struct.ensemble_energy": np.array(-10.0),
            "struct.pairing_probs": np.eye(10),
        }
        
        results = self.validator.validate_thermodynamic_features(nan_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("NaN values" in issue for issue in results["issues"]))
        
        # Test with missing required feature
        missing_features = {
            "struct.mfe": np.array(-10.5),
            "struct.ensemble_energy": np.array(-10.0),
            # Missing pairing_probs
        }
        
        results = self.validator.validate_thermodynamic_features(missing_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("Missing required feature" in issue for issue in results["issues"]))
    
    def test_validate_mi_features(self):
        """Test validation of mutual information features."""
        # Create valid MI features
        valid_features = {
            "mi.scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "mi.coupling_matrix": np.ones((10, 10)) * 0.1,  # 10x10 matrix of 0.1s
        }
        
        results = self.validator.validate_mi_features(valid_features)
        self.assertTrue(results["is_valid"])
        self.assertEqual(len(results["issues"]), 0)
        
        # Create invalid MI features (wrong dimensions)
        invalid_features = {
            "mi.scores": np.array([[0.1, 0.2], [0.3, 0.4]]),  # Should be 1D
            "mi.coupling_matrix": np.ones((10, 10)) * 0.1,
        }
        
        results = self.validator.validate_mi_features(invalid_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("should be 1D" in issue for issue in results["issues"]))
        
        # Create features with negative MI values
        neg_features = {
            "mi.scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "mi.coupling_matrix": np.ones((10, 10)) * -0.1,  # Negative values
        }
        
        results = self.validator.validate_mi_features(neg_features)
        self.assertTrue(results["is_valid"])  # This is a warning, not an error
        self.assertTrue(any("Negative MI values" in warning for warning in results["warnings"]))
        
        # Test with missing required feature
        missing_features = {
            "mi.scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            # Missing coupling_matrix
        }
        
        results = self.validator.validate_mi_features(missing_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("Missing required feature" in issue for issue in results["issues"]))
    
    def test_validate_dihedral_features(self):
        """Test validation of dihedral features."""
        # Create valid dihedral features
        valid_features = {
            "geom.dihedrals": np.random.rand(10, 4) * 2 * np.pi - np.pi,  # 10x4 random dihedrals
        }
        
        results = self.validator.validate_dihedral_features(valid_features)
        self.assertTrue(results["is_valid"])
        self.assertEqual(len(results["issues"]), 0)
        
        # Create invalid dihedral features (wrong dimensions)
        invalid_features = {
            "geom.dihedrals": np.random.rand(10, 3),  # Should be (n, 4)
        }
        
        results = self.validator.validate_dihedral_features(invalid_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("shape (n, 4)" in issue for issue in results["issues"]))
        
        # Create features with out-of-range values
        out_range_features = {
            "geom.dihedrals": np.random.rand(10, 4) * 400 - 200,  # Values outside [-180, 180]
        }
        
        results = self.validator.validate_dihedral_features(out_range_features)
        self.assertTrue(results["is_valid"])  # This is a warning, not an error
        self.assertTrue(any("outside expected range" in warning for warning in results["warnings"]))
        
        # Test with no dihedral features
        no_features = {}
        
        results = self.validator.validate_dihedral_features(no_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("No dihedral features found" in issue for issue in results["issues"]))
    
    def test_validate_feature_compatibility(self):
        """Test validation of feature compatibility."""
        # Create compatible features
        compatible_features = {
            "thermo_features": {
                "struct.mfe": np.array(-10.5),
                "struct.ensemble_energy": np.array(-10.0),
                "struct.pairing_probs": np.eye(10),  # 10x10 identity matrix
            },
            "mi_features": {
                "mi.scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                "mi.coupling_matrix": np.ones((10, 10)) * 0.1,  # 10x10 matrix matching pairing_probs
            },
            "dihedral_features": {
                "geom.dihedrals": np.random.rand(10, 4) * 2 * np.pi - np.pi,  # 10 rows matching sequence length
            },
        }
        
        results = self.validator.validate_feature_compatibility(compatible_features)
        self.assertTrue(results["is_valid"])
        self.assertEqual(len(results["issues"]), 0)
        
        # Create incompatible features (dimension mismatch)
        incompatible_features = {
            "thermo_features": {
                "struct.mfe": np.array(-10.5),
                "struct.ensemble_energy": np.array(-10.0),
                "struct.pairing_probs": np.eye(10),  # 10x10 identity matrix
            },
            "mi_features": {
                "mi.scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                "mi.coupling_matrix": np.ones((15, 15)) * 0.1,  # 15x15 matrix not matching pairing_probs
            },
        }
        
        results = self.validator.validate_feature_compatibility(incompatible_features)
        self.assertFalse(results["is_valid"])
        self.assertTrue(any("does not match sequence length" in issue for issue in results["issues"]))
        
        # Test with missing feature types
        missing_types = {
            "thermo_features": {
                "struct.mfe": np.array(-10.5),
                "struct.ensemble_energy": np.array(-10.0),
                "struct.pairing_probs": np.eye(10),
            },
            # Missing mi_features
        }
        
        results = self.validator.validate_feature_compatibility(missing_types)
        self.assertTrue(results["is_valid"])  # This is a warning, not an error
        self.assertTrue(any("Missing feature types" in warning for warning in results["warnings"]))
    
    def test_validate_target_features(self):
        """Test validation of all features for a target."""
        # Test with valid features (mock_data_manager is configured to return valid features)
        results = self.validator.validate_target_features("valid_target")
        self.assertTrue(results["is_valid"])
        self.assertEqual(len(results["issues"]), 0)
        
        # Test when DataManager raises an exception for a feature type
        def load_features_with_error(target_id, feature_type=None, data_dir=None):
            if feature_type == "mi":
                raise ValueError("Test error loading MI features")
            return self.mock_data_manager.load_features(target_id, feature_type, data_dir)
        
        with patch.object(self.mock_data_manager, 'load_features', side_effect=load_features_with_error):
            results = self.validator.validate_target_features("error_target")
            self.assertTrue(results["is_valid"])  # Still valid because thermo features are OK
            self.assertTrue(any("Failed to load MI features" in warning for warning in results["warnings"]))
    
    def test_validate_batch_results(self):
        """Test validation of batch results."""
        # Create mock batch results
        batch_results = {
            "batch_name": "test_batch",
            "total_targets": 3,
            "results": {
                "target1": {"status": "success"},
                "target2": {"status": "success"},
                "target3": {"status": "error", "error": "Test error"},
            },
        }
        
        # Patch validate_target_features to return known results
        def mock_validate_target(target_id, data_dir=None):
            if target_id == "target1":
                return {
                    "target_id": target_id,
                    "is_valid": True,
                    "feature_results": {
                        "thermo_features": {"is_valid": True, "issues": [], "warnings": []},
                        "mi_features": {"is_valid": True, "issues": [], "warnings": []},
                    },
                    "issues": [],
                    "warnings": [],
                }
            else:
                return {
                    "target_id": target_id,
                    "is_valid": False,
                    "feature_results": {
                        "thermo_features": {"is_valid": False, "issues": ["Test issue"], "warnings": []},
                        "mi_features": {"is_valid": True, "issues": [], "warnings": ["Test warning"]},
                    },
                    "issues": ["Thermo: Test issue"],
                    "warnings": ["MI: Test warning"],
                }
        
        with patch.object(self.validator, 'validate_target_features', side_effect=mock_validate_target):
            results = self.validator.validate_batch_results(batch_results)
            self.assertEqual(results["valid_targets"], 1)
            self.assertEqual(results["invalid_targets"], 1)
            self.assertEqual(results["targets_with_warnings"], 1)
            
            # Check that issues and warnings are properly aggregated
            self.assertTrue("Test issue" in results["summary"]["issues_by_type"]["thermo_features"])
            self.assertTrue("Test warning" in results["summary"]["warnings_by_type"]["mi_features"])
    
    def test_generate_validation_report(self):
        """Test generation of validation report."""
        # Add some validation results first
        self.validator.validation_results["targets"] = {
            "target1": {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "feature_results": {},
            },
            "target2": {
                "is_valid": False,
                "issues": ["Thermo: Test issue"],
                "warnings": ["MI: Test warning"],
                "feature_results": {},
            },
        }
        
        self.validator.validation_results["thermo_features"] = {
            "target1": {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "stats": {"mfe": -10.5, "ensemble_energy": -10.0, "entropy": 0.82},
            },
            "target2": {
                "is_valid": False,
                "issues": ["Test issue"],
                "warnings": [],
                "stats": {"mfe": -9.5, "ensemble_energy": -10.0, "entropy": 0.82},
            },
        }
        
        # Generate report
        report = self.validator.generate_validation_report()
        
        # Check report structure
        self.assertEqual(report["summary"]["total_targets"], 2)
        self.assertEqual(report["summary"]["valid_targets"], 1)
        self.assertEqual(report["summary"]["targets_with_issues"], 1)
        
        # Check with output file
        output_file = os.path.join(self.temp_dir, "validation_report.json")
        report = self.validator.generate_validation_report(output_file=output_file)
        
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            saved_report = json.load(f)
            self.assertEqual(saved_report["summary"]["total_targets"], 2)

if __name__ == "__main__":
    unittest.main()