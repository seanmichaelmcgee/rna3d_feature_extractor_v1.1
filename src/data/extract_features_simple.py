#!/usr/bin/env python3
"""
Modular RNA Feature Extraction CLI Tool

This script provides a streamlined command-line interface to extract RNA features
using the improved thermodynamic_analysis module. This ensures thermodynamic
consistency and automatic propagation of any future improvements to the core module.

Features are saved in NumPy's NPZ format without any validation or JSON serialization
that could cause errors. The tool focuses ONLY on feature extraction; use the 
companion 'visualize_features.py' tool for visualization of the extracted features.

Output files are saved to ./data/processed/features by default, or a custom directory
can be specified.
"""

import os
import sys
import argparse
import time
import traceback
from pathlib import Path

# Import the core thermodynamic analysis module
try:
    # Use relative import since both modules are in the src directory
    from ..analysis import thermodynamic_analysis as thermo
except ImportError:
    try:
        # Alternative approach: add project root to path
        import sys
        from pathlib import Path
        
        # Navigate from src/data up to the project root
        project_root = Path(__file__).parent.parent.parent
        sys.path.append(str(project_root))
        
        # Now import using the full path
        from src.analysis import thermodynamic_analysis as thermo
    except ImportError:
        print("ERROR: Cannot import thermodynamic_analysis module.")
        print("Make sure the file 'src/analysis/thermodynamic_analysis.py' exists.")
        sys.exit(1)
# Import NumPy for file saving
try:
    import numpy as np
    has_numpy = True
except ImportError:
    has_numpy = False
    print("ERROR: NumPy is required for this tool")
    sys.exit(1)

# Test sequences for quick demonstration
TEST_SEQUENCES = {
    'hairpin': 'GGGAAACCC',
    'tRNA': 'GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA',
    'ribozyme': 'GGCGAGGAGCGCUGUUACGUUUCGACAUUCUGAGGACCGGAUGAUGGAUGAUCCCGAUGCUGAUUCGCAGGCGGAUUUCGCGA',
    '5S_rRNA': 'UGCCUGGCGGCCGUAGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAACGCCGUAGCGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGGGAACUGCCAGGCAU'
}

def extract_features(sequence, pf_scale=1.5):
    """
    Extract thermodynamic features using the core thermodynamic_analysis module.
    
    Parameters:
    -----------
    sequence : str
        RNA sequence
    pf_scale : float, optional
        Scaling factor for partition function calculations.
        Higher values (1.5-3.0) help prevent numeric overflow with longer sequences.
        
    Returns:
    --------
    dict
        Dictionary of features
    """
    print(f"Extracting features for sequence of length {len(sequence)}...")
    
    # Use the core module to extract comprehensive features
    print(f"Using partition function scale factor: {pf_scale}")
    features = thermo.extract_thermodynamic_features(sequence, pf_scale=pf_scale)
    
    # Add basic sequence information
    features['sequence'] = sequence
    features['length'] = len(sequence)
    
    # Rename some keys for compatibility with downstream tools
    key_mapping = {
        'mfe': 'mfe',
        'mfe_structure': 'structure',
        'ensemble_energy': 'ensemble_energy',
        'raw_ensemble_energy': 'raw_ensemble_energy',  # Include raw ensemble energy
        'mfe_probability': 'prob_of_mfe',
        'positional_entropy': 'position_entropy',
        'base_pair_probs': 'pairing_probs'
    }
    
    for new_key, old_key in key_mapping.items():
        if old_key in features and new_key != old_key:
            features[new_key] = features[old_key]
    
    # Ensure all expected keys exist
    required_keys = ['mfe', 'structure', 'ensemble_energy', 'raw_ensemble_energy', 'prob_of_mfe', 
                     'position_entropy', 'pairing_probs']
    
    for key in required_keys:
        if key not in features:
            print(f"Warning: Missing expected feature '{key}'. Adding placeholder.")
            if key in ['position_entropy']:
                features[key] = np.zeros(len(sequence))
            elif key in ['pairing_probs']:
                features[key] = np.zeros((len(sequence), len(sequence)))
            elif key in ['structure']:
                features[key] = '.' * len(sequence)
            elif key in ['prob_of_mfe']:
                features[key] = 1.0
            else:
                features[key] = 0.0

    # --- New Addition: Compute Accessibility ---
    if 'pairing_probs' in features:
        pairing_matrix = features['pairing_probs']
        # Compute per-nucleotide pairing probability sum (assuming the diagonal is zero)
        accessibility = 1.0 - np.sum(pairing_matrix, axis=1)
        features['accessibility'] = accessibility

    # --- New Addition: Average Base Pairing Distance ---
    if 'pairing_probs' in features:
        pairing_matrix = features['pairing_probs']
        sequence_length = len(sequence)
        avg_distances = np.zeros(sequence_length)
        for i in range(sequence_length):
            # Exclude self pairing by skipping j == i
            weights = np.array([pairing_matrix[i, j] for j in range(sequence_length) if j != i])
            distances = np.array([abs(i - j) for j in range(sequence_length) if j != i])
            if weights.sum() > 0:
                avg_distances[i] = np.dot(weights, distances) / weights.sum()
            else:
                avg_distances[i] = 0.0
        features['avg_pair_distance_mean'] = np.mean(avg_distances)
        features['avg_pair_distance_std'] = np.std(avg_distances)

    # --- New Addition: Free Energy per Nucleotide ---
    if 'mfe' in features and len(sequence) > 0:
        features['free_energy_per_nucleotide'] = features['mfe'] / len(sequence)
    else:
        features['free_energy_per_nucleotide'] = 0.0

    # --- New Addition: Variability in Local Accessibility ---
    if 'accessibility' in features:
        features['accessibility_mean'] = np.mean(features['accessibility'])
        features['accessibility_variance'] = np.var(features['accessibility'])
    
    print(f"Extracted {len(features)} features")
    return features

def save_features_npz(features, output_file):
    """
    Save features to NPZ file, avoiding JSON serialization completely.
    
    Parameters:
    -----------
    features : dict
        Dictionary of features
    output_file : str or Path
        Output file path
        
    Returns:
    --------
    bool
        Success flag
    """
    try:
        # Convert all numpy types to native python types for problematic fields
        save_dict = {}
        for key, value in features.items():
            # Handle arrays specially
            if isinstance(value, np.ndarray):
                save_dict[key] = value  # Keep arrays as NumPy
            elif isinstance(value, np.bool_):
                save_dict[key] = bool(value)  # Convert NumPy bool to Python bool
            elif isinstance(value, np.integer):
                save_dict[key] = int(value)  # Convert NumPy int to Python int
            elif isinstance(value, np.floating):
                save_dict[key] = float(value)  # Convert NumPy float to Python float
            else:
                save_dict[key] = value  # Keep others as is
        
        # Save to NPZ using savez_compressed
        np.savez_compressed(output_file, **save_dict)
        print(f"Saved features to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error saving features to NPZ: {e}")
        traceback.print_exc()
        return False

def process_sequence(seq_id, sequence, output_dir, verbose=False, pf_scale=1.5):
    """
    Process a single RNA sequence and extract features.
    
    Parameters:
    -----------
    seq_id : str
        Identifier for the RNA sequence
    sequence : str
        RNA sequence to process
    output_dir : Path
        Directory to save output files
    verbose : bool
        Whether to print detailed progress messages
    pf_scale : float, optional
        Scaling factor for partition function calculations.
        Higher values (1.5-3.0) help prevent numeric overflow with longer sequences.
        
    Returns:
    --------
    dict
        Processing result
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Output file path
    output_file = output_dir / f"{seq_id}_features.npz"
    
    # Time tracking
    start_time = time.time()
    
    # Status message
    print(f"Processing {seq_id} (length: {len(sequence)})")
    
    # Extract features
    try:
        # Extract features using the core module
        features = extract_features(sequence, pf_scale=pf_scale)
        
        # Save to NPZ file
        save_success = save_features_npz(features, output_file)
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        
        # Create result summary
        result = {
            'id': seq_id,
            'length': len(sequence),
            'output_file': str(output_file),
            'elapsed_time': elapsed_time,
            'features_extracted': len(features),
            'save_success': save_success,
            'status': 'success' if save_success else 'error'
        }
        
        if verbose:
            print(f"Completed {seq_id} in {elapsed_time:.2f} seconds")
            print(f"Features extracted: {len(features)}")
            print(f"Save success: {save_success}")
            print(f"MFE: {features.get('mfe', 'N/A')}")
            print(f"Raw ensemble energy: {features.get('raw_ensemble_energy', 'N/A')}")
            print(f"Ensemble energy (clamped): {features.get('ensemble_energy', 'N/A')}")
            print(f"MFE probability: {features.get('prob_of_mfe', 'N/A')}")
            # Display new feature statistics
            print(f"Free energy per nucleotide: {features.get('free_energy_per_nucleotide', 'N/A')}")
            print(f"Avg base pair distance (mean): {features.get('avg_pair_distance_mean', 'N/A')}")
            print(f"Avg base pair distance (std): {features.get('avg_pair_distance_std', 'N/A')}")
            print(f"Accessibility (mean): {features.get('accessibility_mean', 'N/A')}")
            print(f"Accessibility variance: {features.get('accessibility_variance', 'N/A')}")
        
        return result
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error processing {seq_id}: {e}")
        if verbose:
            traceback.print_exc()
        
        return {
            'id': seq_id,
            'length': len(sequence),
            'elapsed_time': elapsed_time,
            'status': 'error',
            'error_message': str(e)
        }

def load_sequences_from_csv(csv_path, id_col='id', seq_col='sequence', limit=None):
    """Load RNA sequences from a CSV file."""
    try:
        import pandas as pd
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found in CSV. Available columns: {', '.join(df.columns)}")
        if seq_col not in df.columns:
            raise ValueError(f"Sequence column '{seq_col}' not found in CSV. Available columns: {', '.join(df.columns)}")
        
        # Extract sequences
        sequences = {}
        for _, row in df.iterrows():
            if limit is not None and len(sequences) >= limit:
                break
            sequences[str(row[id_col])] = str(row[seq_col])
        
        print(f"Loaded {len(sequences)} sequences from {csv_path}")
        return sequences
    
    except ImportError:
        print("Error: Pandas is required to load CSV files")
        return {}
    except Exception as e:
        print(f"Error loading sequences from CSV: {e}")
        traceback.print_exc()
        return {}

def batch_process_sequences(sequences, output_dir, verbose=False, pf_scale=1.5):
    """Process multiple RNA sequences in batch mode."""
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Track overall statistics
    batch_start_time = time.time()
    results = []
    success_count = 0
    error_count = 0
    
    # Process each sequence
    seq_count = len(sequences)
    for i, (seq_id, sequence) in enumerate(sequences.items()):
        print(f"Processing {i+1}/{seq_count}: {seq_id}")
        
        result = process_sequence(
            seq_id=seq_id,
            sequence=sequence,
            output_dir=output_dir,
            verbose=verbose,
            pf_scale=pf_scale
        )
        
        results.append(result)
        
        # Update statistics
        if result['status'] == 'success':
            success_count += 1
        else:
            error_count += 1
    
    # Calculate overall statistics
    total_time = time.time() - batch_start_time
    avg_time = total_time / seq_count if seq_count > 0 else 0
    
    # Print summary
    print("\nBatch Processing Summary:")
    print(f"- Total sequences: {seq_count}")
    print(f"- Successful: {success_count}")
    print(f"- Failed: {error_count}")
    print(f"- Total time: {total_time:.2f} seconds")
    print(f"- Average time per sequence: {avg_time:.2f} seconds")
    
    return results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="RNA feature extraction using the thermodynamic_analysis module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('-s', '--sequence', help="RNA sequence to process")
    input_group.add_argument('-i', '--id', default="RNA", help="Identifier for the sequence")
    input_group.add_argument('-c', '--csv', help="Path to CSV file containing RNA sequences")
    input_group.add_argument('--id-col', default="id", help="Column name for sequence IDs in CSV")
    input_group.add_argument('--seq-col', default="sequence", help="Column name for sequences in CSV")
    input_group.add_argument('--limit', type=int, help="Maximum number of sequences to process from CSV")
    input_group.add_argument('-t', '--test', action='store_true', help="Use built-in test sequences")
    input_group.add_argument('--check', action='store_true', help="Run thermodynamic consistency check before processing")
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('-o', '--output-dir', default="./data/processed/features", 
                             help="Directory to save output files")
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--verbose', action='store_true', help="Print detailed progress messages")
    proc_group.add_argument('--pf-scale', type=float, default=1.5, 
                           help="Partition function scaling factor (higher values like 1.5-3.0 for long sequences)")
    
    args = parser.parse_args()
    
    # Check for NumPy
    if not has_numpy:
        print("ERROR: NumPy is required for this tool")
        sys.exit(1)
    
    # Convert output directory to Path and ensure it exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine the source of sequences
    sequences = {}
    
    if args.test:
        # Use built-in test sequences
        sequences = TEST_SEQUENCES
        print(f"Using {len(sequences)} built-in test sequences")
    
    elif args.csv:
        # Load sequences from CSV file
        sequences = load_sequences_from_csv(
            args.csv,
            id_col=args.id_col,
            seq_col=args.seq_col,
            limit=args.limit
        )
        if not sequences:
            print("No sequences loaded from CSV. Exiting.")
            sys.exit(1)
    
    elif args.sequence:
        # Use single sequence provided as argument
        sequences = {args.id: args.sequence}
        print(f"Using single sequence: {args.id} (length: {len(args.sequence)})")
    
    else:
        # No input provided
        parser.print_help()
        print("\nERROR: You must provide a sequence, a CSV file, or use --test option.")
        sys.exit(1)
    
    # Run thermodynamic consistency check if requested
    if args.check and args.sequence:
        test_results = thermo.test_thermodynamic_consistency(args.sequence)
        print("\nThermodynamic Consistency Check Results:")
        for key, value in test_results.items():
            if key != 'error':
                print(f"- {key}: {value}")
    
    # Process the sequences
    if len(sequences) == 1:
        # Single sequence processing
        seq_id, sequence = next(iter(sequences.items()))
        process_sequence(
            seq_id=seq_id,
            sequence=sequence,
            output_dir=output_dir,
            verbose=args.verbose,
            pf_scale=args.pf_scale
        )
    else:
        # Batch processing
        batch_process_sequences(
            sequences=sequences,
            output_dir=output_dir,
            verbose=args.verbose,
            pf_scale=args.pf_scale
        )

if __name__ == "__main__":
    main()