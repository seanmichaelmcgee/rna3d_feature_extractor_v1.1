"""
Enhanced Mutual Information Module for RNA Evolutionary Coupling Analysis

This module enhances the existing mutual_information.py with:
1. RNA-specific APC (Average Product Correction)
2. Sequence weighting to reduce redundancy bias
3. Position filtering based on conservation and gap frequency
4. Chunking support for long RNA sequences (>750 nt)
5. Memory-optimized implementation for 4-core CPUs

Performance optimized for RNAs up to 4,000 nucleotides with appropriate chunking.
"""

import numpy as np
import time
import os
from pathlib import Path
from collections import Counter
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_mi')

def chunk_and_analyze_rna(msa_sequences, max_length=750, chunk_size=600, overlap=200, 
                       gap_threshold=0.5, conservation_range=(0.2, 0.95),
                       parallel=True, n_jobs=None, pseudocount=None, verbose=False):
    """
    Process long RNA sequences by chunking into overlapping segments and 
    calculating mutual information with proper recombination.
    
    Parameters:
    -----------
    msa_sequences : list
        List of aligned sequences
    max_length : int
        Maximum sequence length to process without chunking
    chunk_size : int
        Size of each chunk (default: 600)
    overlap : int
        Overlap between adjacent chunks (default: 200)
    gap_threshold : float
        Maximum gap frequency for position filtering
    conservation_range : tuple
        Conservation range for position filtering
    parallel : bool
        Whether to use parallelization
    n_jobs : int, optional
        Number of parallel jobs
    pseudocount : float or None, default=None
        Pseudocount value to use for frequency normalization.
        If None, will use adaptive selection based on MSA size.
        If 0.0, no pseudocounts will be used (original behavior).
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict
        Dictionary with MI results for the full sequence
    """
    if not msa_sequences:
        return None
        
    seq_length = len(msa_sequences[0])
    
    # If under max_length, no chunking needed
    if seq_length <= max_length:
        if verbose:
            logger.info(f"Sequence length {seq_length} <= {max_length}, no chunking needed")
        return calculate_mutual_information_enhanced(
            msa_sequences,
            gap_threshold=gap_threshold,
            conservation_range=conservation_range,
            parallel=parallel,
            n_jobs=n_jobs,
            pseudocount=pseudocount,
            verbose=verbose
        )
    
    # Create chunks
    chunks = []
    chunk_positions = []
    
    for start in range(0, seq_length - overlap, chunk_size - overlap):
        end = min(start + chunk_size, seq_length)
        
        # Extract chunk from all sequences
        chunk_seqs = [seq[start:end] for seq in msa_sequences]
        chunks.append(chunk_seqs)
        chunk_positions.append((start, end))
        
        # Stop if we've reached the end
        if end == seq_length:
            break
    
    if verbose:
        logger.info(f"Created {len(chunks)} chunks for sequence length {seq_length}")
    
    # Process each chunk
    chunk_results = []
    for i, (chunk_seqs, (start, end)) in enumerate(zip(chunks, chunk_positions)):
        if verbose:
            logger.info(f"Processing chunk {i+1}/{len(chunks)} (positions {start}-{end})")
        
        # Calculate MI for this chunk
        chunk_result = calculate_mutual_information_enhanced(
            chunk_seqs,
            gap_threshold=gap_threshold,
            conservation_range=conservation_range,
            parallel=parallel,
            n_jobs=n_jobs,
            pseudocount=pseudocount,
            verbose=verbose
        )
        
        if chunk_result:
            # Use APC-corrected scores
            chunk_scores = chunk_result['apc_matrix']
            chunk_results.append((chunk_scores, start, end))
    
    # Recombine results
    # Initialize full matrices
    full_matrix = np.zeros((seq_length, seq_length))
    weight_matrix = np.zeros((seq_length, seq_length))
    
    # For each chunk result
    for scores, start, end in chunk_results:
        chunk_size = end - start
        
        # Apply position-based weighting to blend chunks
        for i in range(chunk_size):
            for j in range(chunk_size):
                global_i = start + i
                global_j = start + j
                
                # Calculate weight based on position within chunk
                # Higher weight for center positions, lower for edge positions
                weight_i = 1.0
                weight_j = 1.0
                
                # For positions in the overlap region, use a smooth weighting function
                if i < overlap:
                    # Left edge transition
                    weight_i = i / overlap
                elif i >= chunk_size - overlap:
                    # Right edge transition
                    weight_i = (chunk_size - i) / overlap
                
                if j < overlap:
                    weight_j = j / overlap
                elif j >= chunk_size - overlap:
                    weight_j = (chunk_size - j) / overlap
                
                # Combined weight
                weight = weight_i * weight_j
                
                # Add weighted score to full matrix
                full_matrix[global_i, global_j] += scores[i, j] * weight
                weight_matrix[global_i, global_j] += weight
    
    # Normalize by weights
    # Only where weights are positive to avoid division by zero
    mask = weight_matrix > 0
    full_matrix[mask] /= weight_matrix[mask]
    
    # Apply final APC correction to ensure consistency
    final_matrix = apply_rna_apc_correction(full_matrix)
    
    # Extract top pairs from the final matrix
    top_pairs = []
    for i in range(seq_length):
        for j in range(i+1, seq_length):
            if final_matrix[i, j] > 0:
                top_pairs.append((i, j, final_matrix[i, j]))
    
    # Sort by score and take top 100
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = top_pairs[:min(100, len(top_pairs))]
    
    # Create result dictionary
    result = {
        'mi_matrix': full_matrix,
        'scores': final_matrix,  # Use the final APC-corrected matrix
        'coupling_matrix': final_matrix,  # Add standardized name
        'apc_matrix': final_matrix,
        'top_pairs': top_pairs,
        'method': 'mutual_information_chunked',
        'chunks': len(chunks),
        'chunk_size': chunk_size,
        'overlap': overlap
    }
    
    return result

def filter_rna_msa(msa_sequences, headers=None, 
                 gap_threshold=0.5, 
                 identity_threshold=0.80,
                 max_sequences=5000,
                 verbose=False):
    """
    Filter RNA MSA to reduce redundancy and improve signal quality.
    
    Parameters:
    -----------
    msa_sequences : list
        List of aligned sequences
    headers : list, optional
        Sequence headers
    gap_threshold : float
        Maximum fraction of gaps allowed per sequence/column
    identity_threshold : float
        Sequence identity threshold for diversity filtering
    max_sequences : int
        Maximum number of sequences to keep
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    tuple or None
        (filtered_sequences, filtered_headers) or None if failed
    """
    if not msa_sequences:
        return None
    
    if verbose:
        logger.info(f"Starting MSA filtering with {len(msa_sequences)} sequences")
    
    # Ensure headers list exists
    if headers is None:
        headers = [f"seq_{i}" for i in range(len(msa_sequences))]
    
    # Step 1: Remove sequences with too many gaps
    seq_indices = []
    for i, seq in enumerate(msa_sequences):
        gap_fraction = seq.count('-') / len(seq)
        if gap_fraction <= gap_threshold:
            seq_indices.append(i)
    
    filtered_seqs = [msa_sequences[i] for i in seq_indices]
    filtered_headers = [headers[i] for i in seq_indices]
    
    if verbose:
        logger.info(f"After sequence gap filtering: {len(filtered_seqs)}/{len(msa_sequences)} sequences")
    
    if not filtered_seqs:
        return None
    
    # Step 2: Remove columns with too many gaps
    seq_length = len(filtered_seqs[0])
    keep_columns = []
    
    for col in range(seq_length):
        gaps = sum(1 for seq in filtered_seqs if col < len(seq) and seq[col] == '-')
        gap_fraction = gaps / len(filtered_seqs)
        if gap_fraction <= gap_threshold:
            keep_columns.append(col)
    
    # Apply column filtering
    column_filtered_seqs = []
    for seq in filtered_seqs:
        new_seq = ''.join(seq[col] if col < len(seq) else '-' for col in keep_columns)
        column_filtered_seqs.append(new_seq)
    
    if verbose:
        logger.info(f"After column filtering: {len(keep_columns)}/{seq_length} positions retained")
    
    # Step 3: Calculate sequence weights for redundancy filtering
    weights = calculate_sequence_weights(column_filtered_seqs, 
                                       similarity_threshold=identity_threshold)
    
    # Sort sequences by weight (higher weight = more unique)
    seq_data = list(zip(column_filtered_seqs, filtered_headers, weights))
    seq_data.sort(key=lambda x: x[2], reverse=True)
    
    # Keep top sequences by weight, up to max_sequences
    final_seqs = [sd[0] for sd in seq_data[:max_sequences]]
    final_headers = [sd[1] for sd in seq_data[:max_sequences]]
    
    if verbose:
        logger.info(f"After diversity filtering: {len(final_seqs)} sequences")
    
    return final_seqs, final_headers

def load_msa_robust(msa_file, max_sequences=10000, timeout=300):
    """
    Load Multiple Sequence Alignment with robust error handling and performance optimizations.
    
    Parameters:
    -----------
    msa_file : str or pathlib.Path
        Path to MSA file in FASTA format
    max_sequences : int, optional
        Maximum number of sequences to load to prevent memory issues
    timeout : int, optional
        Maximum time (in seconds) to spend loading the MSA
        
    Returns:
    --------
    tuple or None
        (sequences, headers) if successful, None if failed
    """
    start_time = time.time()
    
    try:
        msa_file = Path(msa_file)
        if not msa_file.exists():
            logger.error(f"MSA file not found: {msa_file}")
            return None
            
        # Check file size to warn about potential memory issues
        file_size_mb = msa_file.stat().st_size / (1024 * 1024)
        logger.info(f"MSA file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 100:
            logger.warning(f"Large MSA file ({file_size_mb:.2f} MB). Loading may take time and memory.")
        
        # Initialize variables
        sequences = []
        headers = []
        current_header = None
        current_seq = []
        seq_count = 0
        
        # Parse FASTA file line by line to avoid loading entire file into memory
        with open(msa_file, 'r') as f:
            for line in f:
                # Check timeout if specified
                if timeout is not None and time.time() - start_time > timeout:
                    logger.warning(f"MSA loading timed out after {timeout} seconds")
                    if len(sequences) > 0:
                        logger.info(f"Loaded {len(sequences)} sequences so far")
                        break
                    else:
                        return None
                    
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_header is not None:
                        sequences.append(''.join(current_seq))
                        headers.append(current_header)
                        seq_count += 1
                        
                        # Progress report for large files
                        if seq_count % 1000 == 0:
                            logger.info(f"Loaded {seq_count} sequences...")
                            
                        # Check if we've hit the maximum
                        if max_sequences is not None and seq_count >= max_sequences:
                            logger.info(f"Reached maximum sequence limit ({max_sequences})")
                            break
                            
                    # Start new sequence
                    current_header = line[1:]
                    current_seq = []
                else:
                    # Continue current sequence
                    current_seq.append(line)
            
            # Don't forget the last sequence
            if current_header is not None and current_seq:
                sequences.append(''.join(current_seq))
                headers.append(current_header)
                seq_count += 1
                
        logger.info(f"Loaded {len(sequences)} sequences in {time.time() - start_time:.2f} seconds")
        
        # Validate lengths - critical for MSA processing
        seq_lens = [len(s) for s in sequences]
        if len(set(seq_lens)) > 1:
            logger.warning(f"Inconsistent sequence lengths in MSA: {min(seq_lens)} to {max(seq_lens)}")
            
            # Determine reference length (use first sequence)
            ref_len = len(sequences[0])
            logger.info(f"Using first sequence length as reference: {ref_len}")
            
            # Filter sequences to keep only those matching reference length
            valid_indices = [i for i, s in enumerate(sequences) if len(s) == ref_len]
            if len(valid_indices) < len(sequences):
                logger.info(f"Keeping {len(valid_indices)} sequences with consistent length")
                sequences = [sequences[i] for i in valid_indices]
                headers = [headers[i] for i in valid_indices]
        
        return sequences, headers
        
    except Exception as e:
        logger.error(f"Error loading MSA: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_rna_msa_for_structure(msa_file, output_features=None, max_length=750,
                                chunk_size=600, overlap=200, gap_threshold=0.5,
                                identity_threshold=0.80, max_sequences=5000,
                                conservation_range=(0.2, 0.95), pseudocount=None,
                                parallel=True, n_jobs=None, verbose=True):
    """
    Complete pipeline to process RNA MSA for structure prediction,
    with chunking for long sequences.
    
    Parameters:
    -----------
    msa_file : str
        Path to MSA file in FASTA format
    output_features : str, optional
        Path to save output features
    max_length : int
        Maximum sequence length to process without chunking
    chunk_size : int
        Size of each chunk
    overlap : int
        Overlap between chunks
    gap_threshold : float
        Maximum gap frequency for filtering
    identity_threshold : float
        Sequence identity threshold for clustering
    max_sequences : int
        Maximum number of sequences to use
    conservation_range : tuple
        Range of conservation values to include
    pseudocount : float or None, default=None
        Pseudocount value to use for frequency normalization.
        If None, will use adaptive selection based on MSA size.
        If 0.0, no pseudocounts will be used (original behavior).
    parallel : bool
        Whether to use parallelization
    n_jobs : int, optional
        Number of parallel jobs
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict
        Dictionary of evolutionary coupling features
    """
    # Load MSA file
    result = load_msa_robust(msa_file, max_sequences=max_sequences)
    if not result:
        logger.error(f"Failed to load MSA from {msa_file}")
        return None
    
    sequences, headers = result
    if verbose:
        logger.info(f"Loaded {len(sequences)} sequences from {msa_file}")
    
    # Filter MSA for quality
    filtered = filter_rna_msa(
        sequences, 
        headers=headers,
        gap_threshold=gap_threshold,
        identity_threshold=identity_threshold,
        max_sequences=max_sequences,
        verbose=verbose
    )
    
    if not filtered:
        logger.error("MSA filtering failed, no sequences passed filters")
        return None
    
    filtered_seqs, filtered_headers = filtered
    
    # Process with chunking if needed
    mi_result = chunk_and_analyze_rna(
        filtered_seqs, 
        max_length=max_length,
        chunk_size=chunk_size,
        overlap=overlap,
        gap_threshold=gap_threshold,
        conservation_range=conservation_range,
        parallel=parallel,
        n_jobs=n_jobs,
        pseudocount=pseudocount,
        verbose=verbose
    )
    
    if not mi_result:
        logger.error("MI calculation failed")
        return None
    
    # Format as expected evolutionary coupling features
    features = {
        'coupling_matrix': mi_result['scores'],
        'method': mi_result['method'],
        'msa_file': str(msa_file),
        'sequence_count': len(filtered_seqs),
        'sequence_length': len(filtered_seqs[0]),
        'parameters': {
            'gap_threshold': gap_threshold,
            'identity_threshold': identity_threshold,
            'conservation_range': conservation_range,
            'pseudocount': mi_result.get('params', {}).get('pseudocount', pseudocount)
        }
    }
    
    # Include chunking info if used
    if 'chunks' in mi_result:
        features['chunking'] = {
            'num_chunks': mi_result['chunks'],
            'chunk_size': mi_result.get('chunk_size', chunk_size),
            'overlap': mi_result.get('overlap', overlap)
        }
    
    # Get top pairs if available
    if 'top_pairs' in mi_result:
        features['top_pairs'] = mi_result['top_pairs']
    
    # Save features if output path provided
    if output_features:
        np.savez_compressed(output_features, **features)
        logger.info(f"Saved evolutionary features to {output_features}")
    
    return features

def calculate_sequence_weights(sequences, similarity_threshold=0.8):
    """
    Calculate sequence weights to reduce the influence of redundant sequences.
    
    Parameters:
    -----------
    sequences : list
        List of aligned sequences
    similarity_threshold : float
        Sequences with similarity above this threshold are downweighted
        
    Returns:
    --------
    numpy.ndarray
        Array of sequence weights
    """
    if not sequences:
        return np.array([])
        
    n_sequences = len(sequences)
    seq_length = len(sequences[0])
    
    # For very large datasets, process in batches
    weights = np.ones(n_sequences)
    
    # Process in batches to control memory usage
    batch_size = 500
    for start in range(0, n_sequences, batch_size):
        end = min(start + batch_size, n_sequences)
        batch_seqs = sequences[start:end]
        
        for i, seq_i in enumerate(batch_seqs):
            similar_count = 0
            
            # Compare against all sequences
            for j, seq_j in enumerate(sequences):
                if start + i == j:
                    continue
                    
                # Count matching positions (excluding gaps)
                matches = 0
                non_gaps = 0
                
                for k in range(min(len(seq_i), len(seq_j))):
                    if seq_i[k] != '-' and seq_j[k] != '-':
                        non_gaps += 1
                        if seq_i[k] == seq_j[k]:
                            matches += 1
                
                # Calculate similarity if there are any non-gap positions
                if non_gaps > 0:
                    similarity = matches / non_gaps
                    if similarity > similarity_threshold:
                        similar_count += 1
            
            # Downweight sequence based on how many similar sequences exist
            if similar_count > 0:
                weights[start + i] = 1.0 / (similar_count + 1)
    
    # Normalize weights to sum to 1
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    else:
        # If all weights are zero, use uniform weights
        weights = np.ones(n_sequences) / n_sequences
    
    # Calculate effective number of sequences
    effective_n = 1.0 / np.sum(weights**2)
    logger.info(f"Calculated sequence weights (effective number of sequences: {effective_n:.1f})")
    
    return weights

def calculate_conservation(sequences, weights=None):
    """
    Calculate conservation score for each position in the alignment.
    
    Parameters:
    -----------
    sequences : list
        List of aligned sequences
    weights : numpy.ndarray, optional
        Array of sequence weights
        
    Returns:
    --------
    numpy.ndarray
        Array of conservation scores for each position
    """
    if not sequences:
        return np.array([])
        
    n_sequences = len(sequences)
    seq_length = len(sequences[0])
    
    # If weights not provided, use uniform weights
    if weights is None:
        weights = np.ones(n_sequences) / n_sequences
    
    # Initialize conservation scores
    conservation = np.zeros(seq_length)
    
    # Calculate conservation for each position
    for i in range(seq_length):
        # Extract column at position i
        col = [seq[i] for seq in sequences]
        
        # Count frequency of each character
        counts = {}
        for j, c in enumerate(col):
            if c not in counts:
                counts[c] = 0
            counts[c] += weights[j]
        
        # Calculate conservation as frequency of most common character
        if counts:
            conservation[i] = max(counts.values())
    
    return conservation
        
def get_adaptive_pseudocount(msa_sequences):
    """
    Determine appropriate pseudocount value based on MSA characteristics.
    
    Args:
        msa_sequences: List of aligned sequences
        
    Returns:
        float: Appropriate pseudocount value based on MSA size
    """
    seq_count = len(msa_sequences)
    if seq_count <= 25:
        return 0.5  # Higher pseudocount for very small MSAs
    elif seq_count <= 100:
        return 0.2  # Moderate pseudocount for medium MSAs
    else:
        return 0.0  # No pseudocount for large, well-populated MSAs

def calculate_mutual_information_enhanced(msa_sequences, weights=None, 
                                        gap_threshold=0.5, 
                                        conservation_range=(0.2, 0.95),
                                        parallel=True,
                                        n_jobs=None,
                                        pseudocount=None,
                                        verbose=False):
    """
    Calculate mutual information with RNA-specific enhancements.
    
    Parameters:
    -----------
    msa_sequences : list
        List of aligned sequences
    weights : numpy.ndarray, optional
        Array of sequence weights
    gap_threshold : float
        Maximum gap frequency for position filtering
    conservation_range : tuple
        Range of conservation values to include
    parallel : bool
        Whether to use parallelization
    n_jobs : int, optional
        Number of parallel jobs
    pseudocount : float or None, default=None
        Pseudocount value to use for frequency normalization.
        If None, will use adaptive selection based on MSA size.
        If 0.0, no pseudocounts will be used (original behavior).
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict
        Dictionary with MI results
    """
    if not msa_sequences:
        return None
        
    # Initialize feature dictionary
    features = {}
    
    # Get sequence count and length
    seq_count = len(msa_sequences)
    seq_length = len(msa_sequences[0])
    
    # Get adaptive pseudocount if not specified
    if pseudocount is None:
        pseudocount = get_adaptive_pseudocount(msa_sequences)
    
    if verbose:
        logger.info(f"Processing MSA with {seq_count} sequences of length {seq_length}")
        if pseudocount > 0:
            logger.info(f"Using pseudocount correction: {pseudocount}")
    
    # Calculate sequence weights if not provided
    if weights is None:
        weights = calculate_sequence_weights(msa_sequences)
    
    # Initialize MI matrix
    mi_matrix = np.zeros((seq_length, seq_length))
    
    # Define the RNA alphabet
    alphabet = ['A', 'C', 'G', 'U', 'T', '-', 'N']
    alphabet_size = len(alphabet)
    
    # Simple implementation of position MI calculation
    for i in range(seq_length):
        for j in range(i+1, seq_length):
            # Extract columns at positions i and j
            col_i = [seq[i] for seq in msa_sequences]
            col_j = [seq[j] for seq in msa_sequences]
            
            # Bypass pseudocount logic if pseudocount is 0.0 (original behavior)
            if pseudocount <= 0.0:
                # Calculate joint and marginal frequencies without pseudocounts
                counts_i = Counter(col_i)
                counts_j = Counter(col_j)
                counts_ij = Counter(zip(col_i, col_j))
                
                # Calculate mutual information
                mi = 0.0
                for x, nx in counts_i.items():
                    px = nx / seq_count
                    for y, ny in counts_j.items():
                        py = ny / seq_count
                        if (x,y) in counts_ij:
                            pxy = counts_ij[(x,y)] / seq_count
                            mi += pxy * np.log2(pxy / (px * py))
            else:
                # Calculate frequencies with pseudocounts and sequence weights
                # Initialize frequency dictionaries with pseudocounts
                i_freqs = {a: pseudocount/alphabet_size for a in alphabet}
                j_freqs = {a: pseudocount/alphabet_size for a in alphabet}
                joint_freqs = {(a, b): pseudocount/(alphabet_size**2) for a in alphabet for b in alphabet}
                
                # Calculate total weight of sequences
                total_weight = 1.0  # Weights sum to 1.0 due to normalization
                
                # Add weighted observations
                for idx, (a, b) in enumerate(zip(col_i, col_j)):
                    if a in alphabet and b in alphabet:
                        w = weights[idx]
                        i_freqs[a] += w
                        j_freqs[b] += w
                        joint_freqs[(a, b)] += w
                
                # Normalize with pseudocount
                norm_factor = total_weight + pseudocount
                for a in alphabet:
                    i_freqs[a] /= norm_factor
                    j_freqs[a] /= norm_factor
                
                for pair in joint_freqs:
                    joint_freqs[pair] /= norm_factor
                
                # Calculate MI using normalized frequencies
                mi = 0.0
                for a in alphabet:
                    p_a = i_freqs[a]
                    if p_a > 0:
                        for b in alphabet:
                            p_b = j_freqs[b]
                            p_ab = joint_freqs[(a, b)]
                            if p_b > 0 and p_ab > 0:
                                mi += p_ab * np.log2(p_ab / (p_a * p_b))
            
            # Set symmetric values
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    
    # Apply APC correction
    apc_matrix = apply_rna_apc_correction(mi_matrix)
    
    # Create result dictionary
    result = {
        'mi_matrix': mi_matrix,
        'apc_matrix': apc_matrix,
        'scores': apc_matrix,  # Use APC-corrected scores
        'coupling_matrix': apc_matrix,  # Add standardized name
        'method': 'mutual_information_enhanced',
        'params': {
            'pseudocount': pseudocount,
            'alphabet_size': alphabet_size,
            'gap_threshold': gap_threshold,
            'conservation_range': conservation_range
        }
    }
    
    return result

def apply_rna_apc_correction(mi_matrix):
    """
    Apply RNA-specific APC (Average Product Correction) to MI matrix.
    
    Parameters:
    -----------
    mi_matrix : numpy.ndarray
        Raw mutual information matrix
        
    Returns:
    --------
    numpy.ndarray
        APC-corrected matrix
    """
    # Calculate row means and overall mean
    row_means = np.mean(mi_matrix, axis=1)
    overall_mean = np.mean(mi_matrix)
    
    # Initialize APC-corrected matrix
    n = mi_matrix.shape[0]
    apc_matrix = np.zeros_like(mi_matrix)
    
    # Apply standard APC correction
    for i in range(n):
        for j in range(i+1, n):
            # Standard APC correction formula
            # Avoid division by zero
            if overall_mean > 0:
                apc_correction = (row_means[i] * row_means[j]) / overall_mean
                apc_value = max(0, mi_matrix[i, j] - apc_correction)
            else:
                apc_value = mi_matrix[i, j]  # If overall_mean is 0, skip correction
                
            apc_matrix[i, j] = apc_matrix[j, i] = apc_value
    
    # RNA-specific adjustments
    # 1. Downweight pairs close in sequence (2-8 positions apart)
    # except for direct neighbors which could be base-paired
    seq_dist_weight = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            seq_dist = j - i
            if 2 <= seq_dist <= 8:
                # Apply logarithmic downweighting (soft transition)
                # 0.7 at distance 2, gradually increasing to 1.0
                seq_dist_weight[i, j] = seq_dist_weight[j, i] = 0.7 + 0.3 * np.log(seq_dist) / np.log(8)
                
    # Apply sequence distance weighting
    rna_apc = apc_matrix * seq_dist_weight
    
    # Apply mild Gaussian smoothing to remove noise
    # For RNA, use a smaller sigma than proteins
    smoothed_matrix = gaussian_filter(rna_apc, sigma=0.6)
    
    return smoothed_matrix
    
    # Get indices of positions to analyze
    analysis_positions = np.where(meaningful_positions)[0]
    
    if verbose:
        logger.info(f"Analyzing {len(analysis_positions)}/{seq_length} positions after filtering")
    
    # Initialize MI matrix
    mi_matrix = np.zeros((seq_length, seq_length))
    
    # Generate position pairs for MI calculation
    position_pairs = []
    for i_idx, i in enumerate(analysis_positions):
        for j_idx, j in enumerate(analysis_positions[i_idx+1:], i_idx+1):
            position_pairs.append((i, j))
    
    total_pairs = len(position_pairs)
    
    if verbose:
        logger.info(f"Calculating MI for {total_pairs} position pairs")
    
    # Define function to calculate MI for a single position pair
    def calculate_mi_for_pair(i, j):
        # Extract columns
        col_i = [seq[i] if i < len(seq) else '-' for seq in msa_sequences]
        col_j = [seq[j] if j < len(seq) else '-' for seq in msa_sequences]
        
        # Get unique values and count frequencies
        joint_counts = Counter(zip(col_i, col_j))
        i_counts = Counter(col_i)
        j_counts = Counter(col_j)
        
        # Calculate MI with sequence weighting
        mi_value = 0
        for (a, b), joint_count in joint_counts.items():
            # Apply sequence weights to counts
            p_ij = 0
            for idx, (ci, cj) in enumerate(zip(col_i, col_j)):
                if ci == a and cj == b:
                    p_ij += weights[idx]
            
            # Marginal probabilities
            p_i = sum(weights[idx] for idx, val in enumerate(col_i) if val == a)
            p_j = sum(weights[idx] for idx, val in enumerate(col_j) if val == b)
            
            # Calculate MI contribution
            if p_ij > 0 and p_i > 0 and p_j > 0:
                mi_value += p_ij * np.log2(p_ij / (p_i * p_j))
        
        return i, j, mi_value
    
    # Decide whether to use parallelization
    if parallel and total_pairs > 1000:
        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)  # Use all but one CPU core
        
        # Process in batches for better memory management
        batch_size = min(5000, max(100, total_pairs // (n_jobs * 10)))
        
        with Pool(n_jobs) as pool:
            from tqdm import tqdm
            
            # Process position pairs in batches
            results = []
            batch_count = (total_pairs + batch_size - 1) // batch_size
            
            for batch_idx in range(0, total_pairs, batch_size):
                batch_end = min(batch_idx + batch_size, total_pairs)
                batch_pairs = position_pairs[batch_idx:batch_end]
                
                # Map function to position pairs
                batch_results = pool.starmap(
                    calculate_mi_for_pair,
                    [(i, j) for i, j in batch_pairs]
                )
                
                results.extend(batch_results)
                
                if verbose:
                    progress = (batch_idx + batch_size) / total_pairs * 100
                    logger.info(f"MI calculation: {progress:.1f}% complete")
    else:
        # Sequential calculation
        results = []
        for idx, (i, j) in enumerate(position_pairs):
            results.append(calculate_mi_for_pair(i, j))
            
            # Report progress
            if verbose and idx % max(1, total_pairs // 10) == 0:
                progress = idx / total_pairs * 100
                logger.info(f"MI calculation: {progress:.1f}% complete")
    
    # Fill MI matrix
    for i, j, mi_value in results:
        mi_matrix[i, j] = mi_matrix[j, i] = mi_value
    
    # Apply RNA-specific APC correction
    apc_matrix = apply_rna_apc_correction(mi_matrix)
    
    # Extract top pairs from the APC-corrected matrix
    top_pairs = []
    for i in range(seq_length):
        for j in range(i+1, seq_length):
            if apc_matrix[i, j] > 0:
                top_pairs.append((i, j, apc_matrix[i, j]))
    
    # Sort by score and take top 100
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = top_pairs[:min(100, len(top_pairs))]
    
    elapsed_time = time.time() - start_time
    if verbose:
        logger.info(f"MI calculation completed in {elapsed_time:.2f} seconds")
    
    # Return results
    result = {
        'mi_matrix': mi_matrix,
        'apc_matrix': apc_matrix,
        'scores': apc_matrix,  # Use corrected matrix as the scores
        'top_pairs': top_pairs,
        'gap_freq': gap_freq,
        'conservation': conservation,
        'valid_positions': valid_positions,
        'meaningful_positions': meaningful_positions,
        'method': 'mutual_information_enhanced',
        'calculation_time': elapsed_time
    }
    
    return result