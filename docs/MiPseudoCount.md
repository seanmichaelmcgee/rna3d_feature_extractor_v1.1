# MI Pseudocount Implementation

This document provides a unified approach for implementing pseudocount corrections in our mutual information (MI) calculation pipeline for RNA analysis.

## 1. Motivation

Sparse multiple sequence alignments (MSAs) can lead to unreliable estimates of amino acid frequencies and mutual information. Pseudocounts help mitigate this issue by adding small values to observed frequencies, preventing zero probabilities and improving statistical robustness. This is particularly important for our RNA MSAs where ~10% of our dataset has relatively few sequences.

## 2. Core Implementation Features

### Adaptive Pseudocount Selection

```python
def get_adaptive_pseudocount(msa_sequences):
    """
    Determine appropriate pseudocount value based on MSA characteristics.
    
    Args:
        msa_sequences: List of aligned sequences
        
    Returns:
        float: Appropriate pseudocount value
    """
    seq_count = len(msa_sequences)
    if seq_count <= 25:
        return 0.5  # Higher pseudocount for very small MSAs
    elif seq_count <= 100:
        return 0.2  # Moderate pseudocount for medium MSAs
    else:
        return 0.0  # No pseudocount for large, well-populated MSAs
```

### Mathematical Foundation

With pseudocounts, the probabilities should be calculated as:

```
P(a) = (count(a) + α/|A|) / (N + α)
P(b) = (count(b) + α/|A|) / (N + α)
P(a,b) = (count(a,b) + α/|A|²) / (N + α)
```

Where:
- α is the pseudocount parameter
- |A| is the alphabet size (typically 7 for RNA: A, C, G, U, T, -, N)
- N is the effective number of sequences (considering weights)

## 3. Implementation Details

### Basic MI Implementation (mutual_information.py)

```python
def calculate_mutual_information(msa_sequences, pseudocount=None, verbose=False):
    """
    Calculate mutual information between all pairs of positions in an MSA.
    
    Args:
        msa_sequences: List of aligned sequences
        pseudocount: Pseudocount value (float) or None for adaptive selection
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing MI scores and metadata
    """
    # Start timing if verbose
    start_time = time.time()
    
    # Get adaptive pseudocount if not specified
    if pseudocount is None:
        pseudocount = get_adaptive_pseudocount(msa_sequences)
    
    # Skip pseudocount logic if pseudocount is 0.0
    use_pseudocount = (pseudocount > 0.0)
    
    # Define alphabet
    alphabet = set(['A', 'C', 'G', 'U', 'T', '-', 'N'])
    alphabet_size = len(alphabet)
    
    # Rest of implementation with pseudocount corrections...
    # [Implementation follows pattern described in Mathematical Foundation]
    
    # Include parameters in return dict
    return {
        'scores': mi_matrix,
        'coupling_matrix': mi_matrix,
        'method': 'mutual_information',
        'top_pairs': top_pairs,
        'params': {
            'pseudocount': pseudocount,
            'alphabet_size': alphabet_size
        },
        'calculation_time': time.time() - start_time if verbose else None
    }
```

### Enhanced MI Implementation (enhanced_mi.py)

```python
def calculate_mutual_information_enhanced(msa_sequences, weights=None, 
                                         gap_threshold=0.5, 
                                         conservation_range=(0.2, 0.95),
                                         parallel=True,
                                         n_jobs=None,
                                         pseudocount=None,
                                         verbose=False):
    """
    Calculate enhanced mutual information for RNA with sequence weighting,
    pseudocount correction, and RNA-specific APC correction.
    
    Args:
        msa_sequences: List of aligned sequences
        weights: Optional sequence weights
        gap_threshold: Maximum allowed gap frequency
        conservation_range: Range of allowed conservation values
        parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs
        pseudocount: Pseudocount value or None for adaptive selection
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing enhanced MI scores and metadata
    """
    # Adapt pseudocount if not specified
    if pseudocount is None:
        pseudocount = get_adaptive_pseudocount(msa_sequences)
    
    # Skip pseudocount logic if pseudocount is 0.0
    use_pseudocount = (pseudocount > 0.0)
    
    # [Implementation for enhanced MI with pseudocount correction]
    # Key points:
    # 1. Initialize frequencies with pseudocounts
    # 2. Incorporate sequence weighting with pseudocounts
    # 3. Properly normalize considering both pseudocounts and weights
    
    # After MI calculation, apply APC correction as normal
    apc_matrix = apply_rna_apc_correction(mi_matrix)
    
    # Return comprehensive results
    result = {
        'mi_matrix': mi_matrix,
        'apc_matrix': apc_matrix,
        'scores': apc_matrix,
        'top_pairs': top_pairs,
        'gap_freq': gap_freq,
        'conservation': conservation,
        'valid_positions': valid_positions,
        'meaningful_positions': meaningful_positions,
        'method': 'mutual_information_enhanced',
        'calculation_time': elapsed_time,
        'params': {
            'pseudocount': pseudocount,
            'alphabet_size': len(alphabet),
            'gap_threshold': gap_threshold,
            'conservation_range': conservation_range
        }
    }
    
    return result
```

### Update to Configuration (mi_config.py)

```python
# Add pseudocount configuration
DEFAULT_CONFIG = {
    # ...existing parameters...
    'pseudocount': None,  # Use adaptive by default
    'use_adaptive_pseudocount': True,  # Whether to adapt based on MSA size
}

# Add to MSA quality configurations
MSA_QUALITY_CONFIGS = {
    'high_quality': {
        # ...existing parameters...
        'pseudocount': 0.2,  # Lower pseudocount for high-quality MSAs
    },
    'medium_quality': {
        # ...existing parameters...
        'pseudocount': 0.5,  # Default value
    },
    'low_quality': {
        # ...existing parameters...
        'pseudocount': 0.8,  # Higher pseudocount for low-quality MSAs
    }
}
```

## 4. Integration with Sequence Weighting

For sequence weighting integration, special care is needed to properly normalize:

```python
# When using sequence weights with pseudocounts
total_weight = sum(weights) if weights is not None else len(msa_sequences)
norm_factor = total_weight + pseudocount

# Initialize with pseudocounts
i_freqs = {a: pseudocount/alphabet_size for a in alphabet}
j_freqs = {a: pseudocount/alphabet_size for a in alphabet}
joint_freqs = {(a, b): pseudocount/(alphabet_size**2) for a in alphabet for b in alphabet}

# Add weighted observations
for idx, (a, b) in enumerate(zip(col_i, col_j)):
    if a in alphabet and b in alphabet:
        w = weights[idx] if weights is not None else 1.0/len(msa_sequences)
        i_freqs[a] += w
        j_freqs[b] += w
        joint_freqs[(a, b)] += w

# Normalize with pseudocount
for a in i_freqs:
    i_freqs[a] /= norm_factor
    j_freqs[a] /= norm_factor

for pair in joint_freqs:
    joint_freqs[pair] /= norm_factor
```

## 5. Backwards Compatibility Strategy

To ensure backward compatibility:

1. When `pseudocount=0.0`, skip pseudocount calculations entirely to match original behavior
2. Default parameter value is `None`, which enables adaptive selection
3. Add configuration flags in `mi_config.py` to fully control the behavior

## 6. Testing Strategy

We will test with real-world MSAs from our existing dataset instead of synthetic MSAs for more realistic validation:

1. **Test Data Selection**:
   - A sparse MSA (≤25 sequences)
   - A medium-sized MSA (50-100 sequences)
   - A large, well-populated MSA (>100 sequences)
   - An MSA with unusual gap patterns or conservation profiles

2. **Validation Tests**:
   - Compare MI values with and without pseudocounts
   - Verify integration with sequence weighting
   - Confirm compatibility with APC correction
   - Test edge cases (very small MSAs, single sequence MSAs)
   - Benchmark computational performance impact

3. **Technical Validation**:
   - Verify probability distributions sum to 1.0
   - Ensure normalization is mathematically consistent
   - Check that adaptive pseudocount selection works correctly
   - Validate frequency distributions with pseudocounts follow expected patterns

## 7. Implementation Plan

1. Update `mutual_information.py` with pseudocount support
2. Modify `enhanced_mi.py` to incorporate pseudocounts with sequence weighting
3. Update configuration in `mi_config.py`
4. Create test cases with representative MSAs
5. Add detailed documentation in function docstrings
6. Update Jupyter notebooks with examples using pseudocounts

## 8. Expected Outcomes

1. Improved MI estimation for sparse MSAs
2. Reduced sensitivity to dataset variations
3. More robust evolutionary feature calculation
4. Maintained or improved performance for downstream RNA structure prediction tasks