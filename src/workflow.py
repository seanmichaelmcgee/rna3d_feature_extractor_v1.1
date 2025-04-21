"""
RNA 3D Feature Extraction Workflow

This module provides a complete workflow for extracting RNA 3D features,
integrating all the components of the refactored architecture:
- DataManager for data loading and saving
- FeatureExtractor for feature extraction
- BatchProcessor for processing multiple targets
- MemoryMonitor for tracking memory usage
- ResultValidator for validating results

Example usage:
    workflow = RNAFeatureExtractionWorkflow()
    results = workflow.run_extraction("data/targets.txt", 
                                      extract_thermo=True,
                                      extract_mi=True)
"""

import os
import logging
import time
import json
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple

from src.data.data_manager import DataManager
from src.features.feature_extractor import FeatureExtractor
from src.processing.batch_processor import BatchProcessor
from src.analysis.memory_monitor import MemoryMonitor
from src.validation.result_validator import ResultValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rna_workflow")

class RNAFeatureExtractionWorkflow:
    """
    Complete workflow for RNA 3D feature extraction.
    
    This class brings together all components of the refactored architecture to
    provide a complete workflow for extracting RNA 3D features.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "data/processed",
        log_dir: str = "data/processed/logs",
        memory_plot_dir: str = "data/processed/memory_plots",
        validation_report_dir: str = "data/processed/validation_reports",
        max_memory_gb: float = 16.0,
        batch_size: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RNA feature extraction workflow.
        
        Args:
            data_dir: Base directory for data files
            output_dir: Directory for output files
            log_dir: Directory for log files
            memory_plot_dir: Directory for memory usage plots
            validation_report_dir: Directory for validation reports
            max_memory_gb: Maximum memory usage in GB
            batch_size: Number of targets to process in each batch
            config: Additional configuration parameters
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.memory_plot_dir = memory_plot_dir
        self.validation_report_dir = validation_report_dir
        self.max_memory_gb = max_memory_gb
        self.batch_size = batch_size
        self.config = config or {}
        
        # Create output directories
        for directory in [self.output_dir, self.log_dir, self.memory_plot_dir, self.validation_report_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.data_manager = DataManager(base_dir=self.data_dir)
        self.memory_monitor = MemoryMonitor()
        self.feature_extractor = FeatureExtractor()
        self.batch_processor = BatchProcessor(
            data_manager=self.data_manager,
            feature_extractor=self.feature_extractor,
            memory_monitor=self.memory_monitor,
            output_dir=self.output_dir,
            log_dir=self.log_dir,
            max_memory_usage_gb=self.max_memory_gb,
            batch_size=self.batch_size,
        )
        self.result_validator = ResultValidator(
            data_manager=self.data_manager,
            config=self.config.get("validation_config"),
        )
    
    def run_extraction(
        self,
        targets_file: str,
        extract_thermo: bool = True,
        extract_mi: bool = True,
        extract_dihedral: bool = False,
        batch_name: Optional[str] = None,
        validate_results: bool = True,
        resume: bool = False,
        save_memory_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete RNA feature extraction workflow.
        
        Args:
            targets_file: File containing target IDs (one per line)
            extract_thermo: Whether to extract thermodynamic features
            extract_mi: Whether to extract mutual information features
            extract_dihedral: Whether to extract dihedral features
            batch_name: Optional name for this batch processing run
            validate_results: Whether to validate results after extraction
            resume: Whether to resume a previously interrupted job
            save_memory_plots: Whether to save memory usage plots
            
        Returns:
            Dict with results and statistics
        """
        logger.info(f"Starting RNA feature extraction workflow")
        start_time = time.time()
        
        # Start memory monitoring for the whole workflow
        self.memory_monitor.start_monitoring("complete_workflow")
        
        # Generate batch name if not provided
        if not batch_name:
            timestamp = int(time.time())
            batch_name = f"batch_{timestamp}"
        
        # Load target IDs
        logger.info(f"Loading target IDs from {targets_file}")
        with open(targets_file, 'r') as f:
            target_ids = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(target_ids)} target IDs")
        
        # Process targets
        if resume:
            logger.info(f"Resuming batch processing for {batch_name}")
            batch_results = self.batch_processor.resume_batch_processing(
                batch_name=batch_name,
                extract_thermo=extract_thermo,
                extract_mi=extract_mi,
                save_intermediates=True,
            )
        else:
            logger.info(f"Starting batch processing for {len(target_ids)} targets")
            batch_results = self.batch_processor.process_targets(
                target_ids=target_ids,
                extract_thermo=extract_thermo,
                extract_mi=extract_mi,
                save_intermediates=True,
                batch_name=batch_name,
            )
        
        # Validate results if requested
        validation_results = None
        if validate_results:
            logger.info("Validating extraction results")
            validation_results = self.result_validator.validate_batch_results(
                batch_results=batch_results,
                data_dir=self.output_dir,
            )
            
            # Generate validation report
            report_file = os.path.join(
                self.validation_report_dir, f"{batch_name}_validation_report.json"
            )
            validation_report = self.result_validator.generate_validation_report(
                output_file=report_file,
                full_details=False,
            )
            
            logger.info(f"Validation report saved to {report_file}")
            
            # Log validation summary
            logger.info(f"Validation summary:")
            logger.info(f"- Valid targets: {validation_results['valid_targets']}/{validation_results['total_targets']}")
            logger.info(f"- Targets with warnings: {validation_results['targets_with_warnings']}")
        
        # Stop memory monitoring
        peak_memory = self.memory_monitor.stop_monitoring()
        
        # Save memory plot if requested
        if save_memory_plots:
            memory_plot_file = os.path.join(
                self.memory_plot_dir, f"{batch_name}_memory_usage.png"
            )
            self.memory_monitor.plot_memory_usage(
                output_file=memory_plot_file,
                show=False,
                clear_history=True,
            )
            logger.info(f"Memory usage plot saved to {memory_plot_file}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare workflow results
        workflow_results = {
            "batch_name": batch_name,
            "targets_file": targets_file,
            "extract_thermo": extract_thermo,
            "extract_mi": extract_mi,
            "extract_dihedral": extract_dihedral,
            "total_targets": len(target_ids),
            "successful_targets": batch_results["successful_targets"],
            "skipped_targets": batch_results["skipped_targets"],
            "skipped_target_ids": batch_results["skipped_target_ids"],
            "execution_time": execution_time,
            "peak_memory_gb": peak_memory,
        }
        
        if validation_results:
            workflow_results["validation"] = {
                "valid_targets": validation_results["valid_targets"],
                "invalid_targets": validation_results["invalid_targets"],
                "targets_with_warnings": validation_results["targets_with_warnings"],
            }
        
        # Save workflow results
        workflow_results_file = os.path.join(
            self.log_dir, f"{batch_name}_workflow_results.json"
        )
        with open(workflow_results_file, 'w') as f:
            json.dump(workflow_results, f, indent=2)
        
        logger.info(f"Workflow completed in {execution_time:.2f} seconds")
        logger.info(f"Peak memory usage: {peak_memory:.2f} GB")
        logger.info(f"Workflow results saved to {workflow_results_file}")
        
        return workflow_results
    
    def extract_single_target(
        self,
        target_id: str,
        extract_thermo: bool = True,
        extract_mi: bool = True,
        extract_dihedral: bool = False,
        validate_results: bool = True,
        save_memory_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract features for a single target.
        
        Args:
            target_id: Target ID to process
            extract_thermo: Whether to extract thermodynamic features
            extract_mi: Whether to extract mutual information features
            extract_dihedral: Whether to extract dihedral features
            validate_results: Whether to validate results after extraction
            save_memory_plot: Whether to save memory usage plot
            
        Returns:
            Dict with results and statistics
        """
        logger.info(f"Extracting features for target {target_id}")
        start_time = time.time()
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring(f"target_{target_id}")
        
        # Load required data
        sequence = self.data_manager.get_sequence_for_target(target_id)
        msa_sequences = self.data_manager.load_msa_data(target_id) if extract_mi else None
        
        # Extract features
        logger.info(f"Extracting features for {target_id}")
        features = self.feature_extractor.extract_features(
            target_id=target_id,
            sequence=sequence,
            msa_sequences=msa_sequences,
            extract_thermo=extract_thermo,
            extract_mi=extract_mi,
        )
        
        # Save features
        if extract_thermo and "thermo_features" in features:
            thermo_file = os.path.join(
                self.output_dir, "thermo_features", f"{target_id}_thermo_features.npz"
            )
            os.makedirs(os.path.dirname(thermo_file), exist_ok=True)
            self.data_manager.save_features(features["thermo_features"], thermo_file)
            logger.info(f"Thermodynamic features saved to {thermo_file}")
        
        if extract_mi and "mi_features" in features:
            mi_file = os.path.join(
                self.output_dir, "mi_features", f"{target_id}_mi_features.npz"
            )
            os.makedirs(os.path.dirname(mi_file), exist_ok=True)
            self.data_manager.save_features(features["mi_features"], mi_file)
            logger.info(f"MI features saved to {mi_file}")
        
        if extract_dihedral and "dihedral_features" in features:
            dihedral_file = os.path.join(
                self.output_dir, "dihedral_features", f"{target_id}_dihedral_features.npz"
            )
            os.makedirs(os.path.dirname(dihedral_file), exist_ok=True)
            self.data_manager.save_features(features["dihedral_features"], dihedral_file)
            logger.info(f"Dihedral features saved to {dihedral_file}")
        
        # Stop memory monitoring
        peak_memory = self.memory_monitor.stop_monitoring()
        
        # Save memory plot if requested
        if save_memory_plot:
            memory_plot_file = os.path.join(
                self.memory_plot_dir, f"{target_id}_memory_usage.png"
            )
            self.memory_monitor.plot_memory_usage(
                output_file=memory_plot_file,
                show=False,
                clear_history=True,
            )
            logger.info(f"Memory usage plot saved to {memory_plot_file}")
        
        # Validate results if requested
        validation_results = None
        if validate_results:
            logger.info(f"Validating features for {target_id}")
            validation_results = self.result_validator.validate_target_features(
                target_id=target_id,
                data_dir=self.output_dir,
            )
            
            # Log validation results
            logger.info(f"Validation results for {target_id}:")
            logger.info(f"- Valid: {validation_results['is_valid']}")
            logger.info(f"- Issues: {len(validation_results['issues'])}")
            logger.info(f"- Warnings: {len(validation_results['warnings'])}")
            
            if validation_results['issues']:
                for issue in validation_results['issues']:
                    logger.warning(f"  Issue: {issue}")
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    logger.info(f"  Warning: {warning}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare results
        results = {
            "target_id": target_id,
            "extract_thermo": extract_thermo,
            "extract_mi": extract_mi,
            "extract_dihedral": extract_dihedral,
            "features_extracted": list(features.keys()),
            "execution_time": execution_time,
            "peak_memory_gb": peak_memory,
        }
        
        if validation_results:
            results["validation"] = {
                "is_valid": validation_results["is_valid"],
                "issues_count": len(validation_results["issues"]),
                "warnings_count": len(validation_results["warnings"]),
            }
        
        logger.info(f"Feature extraction for {target_id} completed in {execution_time:.2f} seconds")
        logger.info(f"Peak memory usage: {peak_memory:.2f} GB")
        
        return results

def main():
    """Run the workflow from the command line."""
    parser = argparse.ArgumentParser(description="RNA 3D Feature Extraction Workflow")
    
    # Input options
    parser.add_argument("--targets-file", "-t", required=True, help="File containing target IDs (one per line)")
    
    # Feature extraction options
    parser.add_argument("--extract-thermo", "-et", action="store_true", help="Extract thermodynamic features")
    parser.add_argument("--extract-mi", "-em", action="store_true", help="Extract mutual information features")
    parser.add_argument("--extract-dihedral", "-ed", action="store_true", help="Extract dihedral features")
    
    # Batch options
    parser.add_argument("--batch-name", "-b", help="Name for this batch processing run")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume a previously interrupted job")
    parser.add_argument("--no-validation", "-nv", action="store_true", help="Skip result validation")
    parser.add_argument("--no-memory-plots", "-nm", action="store_true", help="Skip memory usage plots")
    
    # Resource options
    parser.add_argument("--max-memory", "-m", type=float, default=16.0, help="Maximum memory usage in GB")
    parser.add_argument("--batch-size", "-bs", type=int, default=10, help="Number of targets to process in each batch")
    
    # Directory options
    parser.add_argument("--data-dir", "-d", default="data", help="Base directory for data files")
    parser.add_argument("--output-dir", "-o", default="data/processed", help="Directory for output files")
    parser.add_argument("--log-dir", "-l", default="data/processed/logs", help="Directory for log files")
    
    args = parser.parse_args()
    
    # Ensure at least one feature type is selected
    if not any([args.extract_thermo, args.extract_mi, args.extract_dihedral]):
        parser.error("At least one feature type must be selected (--extract-thermo, --extract-mi, or --extract-dihedral)")
    
    # Create workflow
    workflow = RNAFeatureExtractionWorkflow(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        max_memory_gb=args.max_memory,
        batch_size=args.batch_size,
    )
    
    # Run extraction
    results = workflow.run_extraction(
        targets_file=args.targets_file,
        extract_thermo=args.extract_thermo,
        extract_mi=args.extract_mi,
        extract_dihedral=args.extract_dihedral,
        batch_name=args.batch_name,
        validate_results=not args.no_validation,
        resume=args.resume,
        save_memory_plots=not args.no_memory_plots,
    )
    
    logger.info(f"Workflow completed successfully")
    logger.info(f"- Total targets: {results['total_targets']}")
    logger.info(f"- Successful targets: {results['successful_targets']}")
    logger.info(f"- Skipped targets: {results['skipped_targets']}")
    logger.info(f"- Execution time: {results['execution_time']:.2f} seconds")
    logger.info(f"- Peak memory usage: {results['peak_memory_gb']:.2f} GB")

if __name__ == "__main__":
    main()