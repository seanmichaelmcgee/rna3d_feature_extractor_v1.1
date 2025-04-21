"""
ResultValidator Module

This module validates feature extraction results for consistency and compatibility.
It ensures features have the correct format, dimensions, and content for downstream processing.
"""

import numpy as np
import json
import logging
from pathlib import Path

class ResultValidator:
    """
    Validates feature extraction results for consistency and compatibility.
    """
    
    def __init__(self, verbose=False):
        """
        Initialize validator with verbosity setting.
        
        Args:
            verbose (bool, optional): Whether to print detailed validation information. Defaults to False.
        """
        self.verbose = verbose
        self.logger = logging.getLogger("ResultValidator")
        
        # Validation results
        self.validation_results = {}
        
    def validate_thermodynamic_features(self, features):
        """
        Validate thermodynamic features.
        
        Args:
            features (dict): Thermodynamic features to validate
            
        Returns:
            dict: Validation results with success flag and details
        """
        results = {
            'feature_type': 'thermodynamic',
            'success': False,
            'details': {}
        }
        
        try:
            # Check if features is None or empty
            if features is None or not features:
                results['details']['error'] = "No features provided"
                return results
                
            # Check required keys
            required_keys = ['mfe', 'ensemble_energy']
            missing_keys = [k for k in required_keys if k not in features]
            
            if missing_keys:
                results['details']['error'] = f"Missing required keys: {missing_keys}"
                return results
                
            # Check thermodynamic constraint (optional)
            mfe = features.get('mfe', 0)
            ensemble_energy = features.get('ensemble_energy', 0)
            
            if ensemble_energy < mfe:
                results['details']['warning'] = "Thermodynamic constraint violated: ensemble_energy < mfe"
                # This is a warning, not an error
                
            # Check for pairing information (optional)
            has_pairing = 'structure' in features or 'pairing_probs' in features
            results['details']['has_pairing'] = has_pairing
            
            # Check matrix dimensions if available
            if 'pairing_probs' in features:
                pairing = np.array(features['pairing_probs'])
                if len(pairing.shape) == 2 and pairing.shape[0] == pairing.shape[1]:
                    results['details']['matrix_shape'] = pairing.shape
                else:
                    results['details']['error'] = f"Invalid pairing matrix shape: {pairing.shape}"
                    return results
                    
            # All checks passed
            results['success'] = True
            
        except Exception as e:
            results['details']['error'] = f"Validation error: {str(e)}"
            if self.verbose:
                self.logger.error(f"Error validating thermodynamic features: {e}")
                
        return results
        
    def validate_mi_features(self, features):
        """
        Validate mutual information features.
        
        Args:
            features (dict): MI features to validate
            
        Returns:
            dict: Validation results with success flag and details
        """
        results = {
            'feature_type': 'mi',
            'success': False,
            'details': {}
        }
        
        try:
            # Check if features is None or empty
            if features is None or not features:
                results['details']['error'] = "No features provided"
                return results
                
            # Check required keys
            required_keys = ['scores', 'coupling_matrix', 'method']
            missing_keys = [k for k in required_keys if k not in features]
            
            if missing_keys:
                results['details']['error'] = f"Missing required keys: {missing_keys}"
                return results
                
            # Check for single sequence optimization
            is_single_sequence = features.get('single_sequence', False)
            results['details']['is_single_sequence'] = is_single_sequence
            
            # Check matrix dimensions
            scores = np.array(features['scores'])
            if len(scores.shape) == 2 and scores.shape[0] == scores.shape[1]:
                results['details']['matrix_shape'] = scores.shape
            else:
                results['details']['error'] = f"Invalid scores matrix shape: {scores.shape}"
                return results
                
            # Check method
            method = features.get('method', '')
            results['details']['method'] = method
            
            # Check for top pairs
            has_top_pairs = 'top_pairs' in features and len(features['top_pairs']) > 0
            results['details']['has_top_pairs'] = has_top_pairs
            
            # All checks passed
            results['success'] = True
            
        except Exception as e:
            results['details']['error'] = f"Validation error: {str(e)}"
            if self.verbose:
                self.logger.error(f"Error validating MI features: {e}")
                
        return results
        
    def validate_feature_compatibility(self, features):
        """
        Validate compatibility with downstream processing.
        
        Args:
            features (dict): Features to validate, containing both thermodynamic and MI features
            
        Returns:
            dict: Validation results with success flag and details
        """
        results = {
            'feature_type': 'combined',
            'success': False,
            'details': {}
        }
        
        try:
            # Check if features is None or empty
            if features is None or not features:
                results['details']['error'] = "No features provided"
                return results
                
            # Check presence of feature types
            has_thermo = 'thermo' in features and features['thermo'] is not None
            has_mi = 'mi' in features and features['mi'] is not None
            
            results['details']['has_thermo'] = has_thermo
            results['details']['has_mi'] = has_mi
            
            # Validate individual feature types
            if has_thermo:
                thermo_results = self.validate_thermodynamic_features(features['thermo'])
                results['details']['thermo_validation'] = thermo_results
                
            if has_mi:
                mi_results = self.validate_mi_features(features['mi'])
                results['details']['mi_validation'] = mi_results
                
            # Check dimensional compatibility
            if has_thermo and has_mi:
                thermo_shape = None
                mi_shape = None
                
                # Get thermodynamic matrix shape
                if 'pairing_probs' in features['thermo']:
                    thermo_shape = np.array(features['thermo']['pairing_probs']).shape
                    
                # Get MI matrix shape
                if 'scores' in features['mi']:
                    mi_shape = np.array(features['mi']['scores']).shape
                    
                # Check if shapes match
                if thermo_shape and mi_shape:
                    shapes_match = thermo_shape == mi_shape
                    results['details']['shapes_match'] = shapes_match
                    
                    if not shapes_match:
                        results['details']['warning'] = f"Matrix shapes don't match: thermo {thermo_shape}, mi {mi_shape}"
                        # This is a warning, not an error
                        
            # Determine overall success
            # At least one feature type must be valid
            thermo_success = has_thermo and results['details'].get('thermo_validation', {}).get('success', False)
            mi_success = has_mi and results['details'].get('mi_validation', {}).get('success', False)
            
            results['success'] = thermo_success or mi_success
            
        except Exception as e:
            results['details']['error'] = f"Validation error: {str(e)}"
            if self.verbose:
                self.logger.error(f"Error validating feature compatibility: {e}")
                
        return results
        
    def validate_target_features(self, target_id, data_manager):
        """
        Validate all features for a specific target.
        
        Args:
            target_id (str): Target ID to validate
            data_manager: DataManager instance for loading features
            
        Returns:
            dict: Validation results
        """
        results = {
            'target_id': target_id,
            'success': False,
            'features_found': [],
            'details': {}
        }
        
        try:
            # Load thermodynamic features
            thermo_features = data_manager.load_features(target_id, 'thermo')
            has_thermo = thermo_features is not None
            
            # Load MI features
            mi_features = data_manager.load_features(target_id, 'mi')
            has_mi = mi_features is not None
            
            if has_thermo:
                results['features_found'].append('thermo')
                thermo_results = self.validate_thermodynamic_features(thermo_features)
                results['details']['thermo'] = thermo_results
                
            if has_mi:
                results['features_found'].append('mi')
                mi_results = self.validate_mi_features(mi_features)
                results['details']['mi'] = mi_results
                
            # Validate compatibility
            if has_thermo and has_mi:
                compatibility = self.validate_feature_compatibility({
                    'thermo': thermo_features,
                    'mi': mi_features
                })
                results['details']['compatibility'] = compatibility
                
            # Determine overall success
            # At least one feature type must be found and valid
            thermo_success = has_thermo and results['details'].get('thermo', {}).get('success', False)
            mi_success = has_mi and results['details'].get('mi', {}).get('success', False)
            
            results['success'] = thermo_success or mi_success
            
            # Store in validation results
            self.validation_results[target_id] = results
            
        except Exception as e:
            results['details']['error'] = f"Validation error: {str(e)}"
            if self.verbose:
                self.logger.error(f"Error validating features for {target_id}: {e}")
                
        return results
        
    def generate_validation_report(self, output_file=None):
        """
        Generate a validation report based on validation results.
        
        Args:
            output_file (str or Path, optional): Path to save the report. Defaults to None.
            
        Returns:
            dict: Summary of validation results
        """
        if not self.validation_results:
            if self.verbose:
                self.logger.warning("No validation results to report")
            return {'error': 'No validation results available'}
            
        try:
            # Calculate overall statistics
            total_targets = len(self.validation_results)
            valid_targets = sum(1 for r in self.validation_results.values() if r.get('success', False))
            
            # Count feature types
            feature_counts = {
                'thermo': sum(1 for r in self.validation_results.values() if 'thermo' in r.get('features_found', [])),
                'mi': sum(1 for r in self.validation_results.values() if 'mi' in r.get('features_found', []))
            }
            
            # Count targets with specific feature combinations
            both_features = sum(1 for r in self.validation_results.values() 
                             if 'thermo' in r.get('features_found', []) and 'mi' in r.get('features_found', []))
            
            # Create summary
            summary = {
                'timestamp': np.datetime64('now').astype(str),
                'total_targets': total_targets,
                'valid_targets': valid_targets,
                'invalid_targets': total_targets - valid_targets,
                'success_rate': valid_targets / total_targets if total_targets > 0 else 0,
                'feature_counts': feature_counts,
                'targets_with_all_features': both_features
            }
            
            # Generate detailed report if output file specified
            if output_file:
                report = {
                    'summary': summary,
                    'target_results': self.validation_results
                }
                
                # Save report to file
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                    
                if self.verbose:
                    self.logger.info(f"Validation report saved to {output_file}")
                    
            return summary
            
        except Exception as e:
            error_msg = f"Error generating validation report: {e}"
            if self.verbose:
                self.logger.error(error_msg)
            return {'error': error_msg}