"""
Mutual Information Module

This module provides a simplified implementation of evolutionary coupling analysis
using mutual information as an alternative to more complex Direct Coupling Analysis (DCA)
methods like plmDCA.

This can be used as a fallback when more intensive methods fail or timeout.
"""

import numpy as np
from collections import Counter
import time
import logging

def calculate_mutual_information(msa_sequences, verbose=False):
    """
    Calculate mutual information between positions in the MSA.
    This is simpler and faster than DCA methods.
    
    Parameters:
    -----------
    msa_sequences : list of str
        List of aligned RNA sequences from an MSA
    verbose : bool, default=False
        Whether to print progress information
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'scores': numpy array of shape (seq_len, seq_len) with MI scores
        - 'coupling_matrix': numpy array of shape (seq_len, seq_len) with MI scores (standardized name)
        - 'method': 'mutual_information'
        - 'top_pairs': list of (i, j, score) tuples for top scoring pairs
    """
    if not msa_sequences:
        if verbose:
            print("No sequences provided")
        return None
        
    # Get dimensions
    n_seqs = len(msa_sequences)
    seq_len = len(msa_sequences[0])
    
    if verbose:
        print(f"Calculating mutual information for {n_seqs} sequences of length {seq_len}")
        start_time = time.time()
    
    # Convert MSA to a numpy array for faster processing
    msa_array = np.array([list(seq) for seq in msa_sequences])
    
    # Initialize MI matrix
    mi_matrix = np.zeros((seq_len, seq_len))
    
    # Define allowed characters (including gap)
    allowed_chars = set(['A', 'C', 'G', 'U', 'T', '-', 'N', 'a', 'c', 'g', 'u', 't', 'n'])
    
    # Calculate MI for each pair of positions
    total_pairs = (seq_len * (seq_len - 1)) // 2
    processed = 0
    
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            # Extract columns
            col_i = msa_array[:, i]
            col_j = msa_array[:, j]
            
            # Calculate frequencies
            chars_i = Counter(col_i)
            chars_j = Counter(col_j)
            joint_counts = Counter(zip(col_i, col_j))
            
            # Calculate MI
            mi = 0
            for ci in set(col_i):
                if ci not in allowed_chars:
                    continue
                    
                p_i = chars_i.get(ci, 0) / n_seqs
                if p_i > 0:
                    for cj in set(col_j):
                        if cj not in allowed_chars:
                            continue
                            
                        p_j = chars_j.get(cj, 0) / n_seqs
                        if p_j > 0:
                            p_ij = joint_counts.get((ci, cj), 0) / n_seqs
                            if p_ij > 0:
                                mi += p_ij * np.log2(p_ij / (p_i * p_j))
            
            mi_matrix[i, j] = mi_matrix[j, i] = mi
            
            # Update progress
            processed += 1
            if verbose and processed % max(1, total_pairs // 10) == 0:
                elapsed = time.time() - start_time
                progress = processed / total_pairs * 100
                remaining = elapsed / processed * (total_pairs - processed) if processed > 0 else 0
                print(f"Processed {processed}/{total_pairs} position pairs ({progress:.1f}%) - "
                      f"Time elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
    
    # Calculate top pairs
    top_pairs = []
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            top_pairs.append((i, j, mi_matrix[i, j]))
    
    # Sort by MI score, highest first
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Keep top 100 or fewer if there aren't that many
    top_pairs = top_pairs[:min(100, len(top_pairs))]
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Mutual information calculation complete in {elapsed:.1f}s")
        print(f"Top MI score: {top_pairs[0][2] if top_pairs else 0}")
    
    # Return the same structure as expected from DCA methods
    # Include coupling_matrix as a standardized name for the MI matrix
    return {
        'scores': mi_matrix,
        'coupling_matrix': mi_matrix,  # Add standardized name
        'method': 'mutual_information',
        'top_pairs': top_pairs,
        'calculation_time': time.time() - start_time if verbose else None
    }

def convert_mi_to_evolutionary_features(mi_result, target_data, output_file=None):
    """
    Convert mutual information results to the expected evolutionary coupling features format.
    
    Parameters:
    -----------
    mi_result : dict
        Output from calculate_mutual_information
    target_data : pandas.DataFrame
        Target structure data
    output_file : str or Path, optional
        Path to save the features
        
    Returns:
    --------
    dict
        Dictionary with coupling features in the expected format
    """
    if mi_result is None:
        return None
    
    # Extract number of residues
    n_residues = len(target_data)
    
    # Extract the MI matrix
    mi_matrix = mi_result['scores']
    
    # Create the features dictionary
    features = {
        'coupling_matrix': mi_matrix,
        'method': mi_result['method'],
        'top_pairs': np.array(mi_result['top_pairs']) if mi_result['top_pairs'] else np.array([])
    }
    
    # Calculate correlation between MI scores and physical distances if possible
    try:
        # Extract coordinates from target_data
        coords = np.array([
            [target_data.iloc[i]['x_1'], target_data.iloc[i]['y_1'], target_data.iloc[i]['z_1']]
            for i in range(n_residues)
        ])
        
        # Calculate pairwise distances
        distances = np.zeros((n_residues, n_residues))
        for i in range(n_residues):
            for j in range(n_residues):
                distances[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
        
        # Calculate correlation between MI scores and distances
        # (exclude diagonal and lower triangle to avoid redundancy)
        mi_values = []
        dist_values = []
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                mi_values.append(mi_matrix[i, j])
                dist_values.append(distances[i, j])
        
        # Convert to numpy arrays
        mi_values = np.array(mi_values)
        dist_values = np.array(dist_values)
        
        # Calculate correlation (negative correlation expected)
        corr = np.corrcoef(mi_values, dist_values)[0, 1]
        features['score_distance_correlation'] = corr
        
    except Exception as e:
        print(f"Could not calculate correlation with distances: {e}")
    
    # Save features if output file provided
    if output_file is not None:
        np.savez_compressed(output_file, **features)
        print(f"Saved coupling features to {output_file}")
    
    return features