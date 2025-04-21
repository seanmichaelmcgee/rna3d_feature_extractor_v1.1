"""
BatchProcessor Module

This module manages batch processing of multiple RNA targets.
It coordinates data loading and feature extraction, handles memory-aware scheduling,
and tracks progress and results.
"""

import time
import json
import logging
from pathlib import Path

class BatchProcessor:
    """
    Manages batch processing of multiple RNA targets.
    """
    
    def __init__(self, data_manager, feature_extractor, 
                 memory_monitor=None, batch_size=5, verbose=False):
        """
        Initialize with required components and settings.
        
        Args:
            data_manager: DataManager instance for data operations
            feature_extractor: FeatureExtractor instance for feature extraction
            memory_monitor: Memory monitoring object. Defaults to None.
            batch_size (int, optional): Number of targets to process in a batch. Defaults to 5.
            verbose (bool, optional): Whether to print detailed progress. Defaults to False.
        """
        self.data_manager = data_manager
        self.feature_extractor = feature_extractor
        self.memory_monitor = memory_monitor
        self.batch_size = batch_size
        self.verbose = verbose
        self.logger = logging.getLogger("BatchProcessor")
        
        # Results tracking
        self.results = {}
        
    def process_target(self, target_id, extract_thermo=True, extract_mi=True):
        """
        Process a single target, extracting requested feature types.
        
        Args:
            target_id (str): Target ID
            extract_thermo (bool, optional): Whether to extract thermodynamic features. Defaults to True.
            extract_mi (bool, optional): Whether to extract MI features. Defaults to True.
            
        Returns:
            dict: Dictionary with results for each feature type
        """
        if self.verbose:
            self.logger.info(f"Processing target: {target_id}")
            
        results = {'target_id': target_id}
        start_time = time.time()
        
        # Load common data that might be used by multiple feature types
        sequence = self.data_manager.get_sequence_for_target(target_id) if extract_thermo else None
        msa_sequences = self.data_manager.load_msa_data(target_id) if extract_mi else None
        
        # Extract thermodynamic features
        if extract_thermo:
            thermo_file = self.data_manager.thermo_dir / f"{target_id}_thermo_features.npz"
            
            if thermo_file.exists():
                if self.verbose:
                    self.logger.info(f"Thermodynamic features already exist for {target_id}")
                results['thermo'] = {'success': True, 'skipped': True}
            else:
                thermo_features = self.feature_extractor.extract_thermodynamic_features(sequence)
                
                if thermo_features:
                    # Add target ID to features
                    thermo_features['target_id'] = target_id
                    
                    # Save features
                    save_success = self.data_manager.save_features(thermo_features, thermo_file)
                    results['thermo'] = {'success': save_success}
                else:
                    results['thermo'] = {'success': False}
        
        # Extract MI features
        if extract_mi:
            mi_file = self.data_manager.mi_dir / f"{target_id}_mi_features.npz"
            
            if mi_file.exists():
                if self.verbose:
                    self.logger.info(f"MI features already exist for {target_id}")
                results['mi'] = {'success': True, 'skipped': True}
            else:
                mi_features = self.feature_extractor.extract_mi_features(msa_sequences)
                
                if mi_features:
                    # Add target ID to features
                    mi_features['target_id'] = target_id
                    
                    # Save features
                    save_success = self.data_manager.save_features(mi_features, mi_file)
                    results['mi'] = {'success': save_success}
                else:
                    results['mi'] = {'success': False}
        
        # Calculate total time
        elapsed_time = time.time() - start_time
        results['elapsed_time'] = elapsed_time
        
        if self.verbose:
            self.logger.info(f"Completed processing {target_id} in {elapsed_time:.2f} seconds")
            
        return results
        
    def batch_process_targets(self, target_ids, extract_thermo=True, extract_mi=True):
        """
        Process multiple targets in batch mode.
        
        Args:
            target_ids (list): List of target IDs
            extract_thermo (bool, optional): Whether to extract thermodynamic features. Defaults to True.
            extract_mi (bool, optional): Whether to extract MI features. Defaults to True.
            
        Returns:
            dict: Dictionary with results for each target
        """
        if self.verbose:
            self.logger.info(f"Starting batch processing for {len(target_ids)} targets")
            
        start_time = time.time()
        
        batch_results = {}
        success_counts = {'thermo': 0, 'mi': 0}
        skipped_counts = {'thermo': 0, 'mi': 0}
        
        # Process targets in batches
        for i in range(0, len(target_ids), self.batch_size):
            batch_end = min(i + self.batch_size, len(target_ids))
            batch = target_ids[i:batch_end]
            
            if self.verbose:
                self.logger.info(f"Processing batch {i//self.batch_size + 1}: targets {i+1}-{batch_end} of {len(target_ids)}")
                
            # Process each target in the batch
            for target_id in batch:
                result = self.process_target(
                    target_id, 
                    extract_thermo=extract_thermo, 
                    extract_mi=extract_mi
                )
                
                batch_results[target_id] = result
                
                # Update success and skip counts
                for feature_type in ['thermo', 'mi']:
                    if feature_type in result:
                        if result[feature_type].get('success', False):
                            success_counts[feature_type] += 1
                        if result[feature_type].get('skipped', False):
                            skipped_counts[feature_type] += 1
            
            # Memory cleanup after batch
            if self.memory_monitor:
                self.memory_monitor.cleanup()
                
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Create summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_targets': len(target_ids),
            'total_time': total_time,
            'success_counts': success_counts,
            'skipped_counts': skipped_counts,
            'target_results': batch_results
        }
        
        # Save summary to results
        self.results = summary
        
        if self.verbose:
            self.logger.info(f"Batch processing complete!")
            self.logger.info(f"Total targets: {len(target_ids)}")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            
            if extract_thermo:
                self.logger.info(f"Thermodynamic features: {success_counts['thermo']} successful ({skipped_counts['thermo']} skipped)")
                
            if extract_mi:
                self.logger.info(f"MI features: {success_counts['mi']} successful ({skipped_counts['mi']} skipped)")
                
        return batch_results
        
    def get_optimal_batch_size(self, target_ids, available_memory):
        """
        Determine optimal batch size based on targets and available memory.
        
        Args:
            target_ids (list): List of target IDs
            available_memory (float): Available memory in GB
            
        Returns:
            int: Optimal batch size
        """
        # Placeholder implementation
        # In practice, this would analyze sequence lengths, MSA sizes, etc.
        
        # Default conservative batch size
        default_batch_size = 5
        
        # Adjust based on available memory
        if available_memory < 4:
            return max(1, default_batch_size // 2)
        elif available_memory > 16:
            return default_batch_size * 2
        else:
            return default_batch_size
            
    def save_summary(self, output_file):
        """
        Save processing summary to a JSON file.
        
        Args:
            output_file (str or Path): Path to save the summary
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            if self.verbose:
                self.logger.info(f"Saved processing summary to {output_file}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
            return False