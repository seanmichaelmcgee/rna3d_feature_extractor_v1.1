"""
FeatureExtractor Module

This module handles extraction of various features from RNA sequences and alignments.
It provides functionality to extract thermodynamic features using ViennaRNA and
mutual information features from MSA data, with optimizations for single-sequence MSAs.
"""

import numpy as np
import time
import logging
from pathlib import Path

class FeatureExtractor:
    """
    Extracts various features from RNA sequences and alignments.
    """
    
    def __init__(self, memory_monitor=None, verbose=False):
        """
        Initialize with optional memory monitoring.
        
        Args:
            memory_monitor: Memory monitoring object. Defaults to None.
            verbose (bool, optional): Whether to print detailed progress. Defaults to False.
        """
        self.memory_monitor = memory_monitor
        self.verbose = verbose
        self.logger = logging.getLogger("FeatureExtractor")
        
        # Default parameters
        self.pf_scale = 1.5  # Partition function scaling factor for ViennaRNA
        
    def extract_thermodynamic_features(self, sequence, pf_scale=None):
        """
        Extract thermodynamic features for an RNA sequence.
        
        Args:
            sequence (str): RNA sequence
            pf_scale (float, optional): Partition function scaling factor. Defaults to None.
            
        Returns:
            dict: Dictionary with thermodynamic features or None if failed
        """
        if pf_scale is None:
            pf_scale = self.pf_scale
            
        if self.verbose:
            self.logger.info(f"Extracting thermodynamic features for sequence of length {len(sequence)}")
            
        try:
            # Track memory if monitor is available
            if self.memory_monitor:
                self.memory_monitor.log_memory_usage(f"Before thermo features (len={len(sequence)})")
                
            # Placeholder for actual feature extraction
            # This would call ViennaRNA or similar functionality
            features = {
                'sequence': sequence,
                'mfe': 0.0,
                'ensemble_energy': 0.0,
                'pf_scale': pf_scale
            }
            
            if self.memory_monitor:
                self.memory_monitor.log_memory_usage(f"After thermo features")
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting thermodynamic features: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def extract_mi_features(self, msa_sequences, pseudocount=None):
        """
        Extract Mutual Information features from MSA sequences.
        
        Args:
            msa_sequences (list): List of MSA sequences
            pseudocount (float, optional): Pseudocount value for MI calculation. Defaults to None.
            
        Returns:
            dict: Dictionary with MI features or None if failed
        """
        if not msa_sequences:
            self.logger.error("No MSA sequences provided")
            return None
            
        try:
            if self.verbose:
                self.logger.info(f"Extracting MI features for {len(msa_sequences)} sequences of length {len(msa_sequences[0])}")
                
            # Track memory if monitor is available
            if self.memory_monitor:
                self.memory_monitor.log_memory_usage(f"Before MI features (seq_len={len(msa_sequences[0])}, msa_size={len(msa_sequences)})")
                
            # Check for single-sequence MSA (optimization)
            unique_sequences = set(msa_sequences)
            if len(unique_sequences) <= 1:
                if self.verbose:
                    self.logger.info(f"Single-sequence MSA detected, optimizing MI calculation")
                
                # For single sequence, return zero matrix with metadata
                seq_len = len(msa_sequences[0])
                mi_matrix = np.zeros((seq_len, seq_len))
                
                features = {
                    'scores': mi_matrix,
                    'coupling_matrix': mi_matrix,
                    'method': 'mutual_information',
                    'top_pairs': [],
                    'single_sequence': True
                }
                
            else:
                # Placeholder for actual MI calculation
                # This would call the mutual information calculation
                seq_len = len(msa_sequences[0])
                mi_matrix = np.zeros((seq_len, seq_len))
                
                features = {
                    'scores': mi_matrix,
                    'coupling_matrix': mi_matrix,
                    'method': 'mutual_information',
                    'top_pairs': [],
                    'single_sequence': False
                }
                
            if self.memory_monitor:
                self.memory_monitor.log_memory_usage(f"After MI features")
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting MI features: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def validate_features(self, features, feature_type):
        """
        Validate extracted features for correctness and format compatibility.
        
        Args:
            features (dict): Features to validate
            feature_type (str): Type of features ('thermo', 'mi')
            
        Returns:
            bool: True if features are valid, False otherwise
        """
        if features is None:
            return False
            
        try:
            if feature_type == 'thermo':
                # Check required keys for thermodynamic features
                required_keys = ['mfe', 'ensemble_energy']
                return all(k in features for k in required_keys)
                
            elif feature_type == 'mi':
                # Check required keys for MI features
                required_keys = ['scores', 'coupling_matrix', 'method']
                return all(k in features for k in required_keys)
                
            else:
                self.logger.error(f"Unknown feature type: {feature_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating features: {e}")
            return False