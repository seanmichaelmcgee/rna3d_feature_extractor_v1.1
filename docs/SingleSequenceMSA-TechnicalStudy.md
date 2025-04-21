# Technical Study: Optimizing Mutual Information Calculation for Single-Sequence MSAs

## 1. Introduction and Problem Statement

### 1.1 Background

Mutual Information (MI) is a key feature used in RNA structure prediction within our RNA 3D feature extractor pipeline. MI quantifies the correlation between positions in a Multiple Sequence Alignment (MSA), providing evolutionary coupling information that helps predict RNA 3D structure.

### 1.2 Problem Definition

A significant optimization opportunity has been identified in our RNA MI pipeline: **the MI calculation is currently performed even when there's only one unique sequence in the MSA**, which is both computationally wasteful and produces meaningless results.

Specifically:
- MI measures correlation between positions across multiple aligned sequences
- With only one sequence, there's no variation to compute correlation from
- For single-sequence MSAs, the coupling matrix contains identical values (e.g., 0.6212779434913056)
- This wastes computational resources, especially for very long RNA sequences (>1000 nt)
- The resulting features don't provide useful information for downstream analysis

### 1.3 Impact

The current implementation:
- Unnecessarily consumes CPU time and memory
- Creates potentially misleading MI matrix values
- Slows down the feature extraction process for large datasets
- May negatively impact downstream model performance due to non-informative features

## 2. Analysis of Current Implementation

### 2.1 Current Call Path

The MI calculation is called from three main feature extraction notebooks:
- `train_features_extraction.ipynb`
- `test_features_extraction.ipynb` 
- `validation_features_extraction.ipynb`

All three notebooks follow a nearly identical pattern:

```python
# Extract MI features for a target
def extract_mi_features_for_target(target_id, structure_data=None, msa_sequences=None):
    # ...
    
    # Check if MSA has at least 2 sequences
    if msa_sequences is None or len(msa_sequences) < 2:
        print(f"Failed to get MSA data for {target_id} or not enough sequences")
        return None
    
    # Calculate MI
    mi_result = calculate_mutual_information(msa_sequences, verbose=VERBOSE)
    
    # Process and save results
    # ...
```

### 2.2 Core Functions

Two main functions are responsible for MI calculation:

1. `calculate_mutual_information` in `mutual_information.py`
2. `calculate_mutual_information_enhanced` in `enhanced_mi.py` 

Neither function currently checks if the MSA contains only one unique sequence (multiple identical copies of the same sequence).

### 2.3 Limitations in Current Approach

The core issue is that while there is a basic check for `len(msa_sequences) < 2` in the notebook code, this doesn't handle the case of multiple identical sequences in the MSA. Additionally, the costly computation still occurs in cases where one unique sequence is repeated multiple times.

## 3. Proposed Solution

### 3.1 Solution Overview

We propose to add early detection and handling of single-sequence MSAs directly in the MI calculation functions. The solution will:

1. Detect when an MSA has only one unique sequence (either exactly one sequence or multiple identical sequences)
2. Skip the actual MI calculation in this case
3. Generate appropriate "dummy" MI features with metadata indicating this situation
4. Ensure consistent implementation across both MI calculation functions
5. Maintain API compatibility for downstream processes

### 3.2 Detection Approach

To determine if an MSA contains only one unique sequence:

```python
# Convert sequences to a set to get unique sequences
unique_sequences = set(msa_sequences)
if len(unique_sequences) <= 1:
    # Only one unique sequence - return dummy result
    # ...
```

### 3.3 Output Structure

To maintain compatibility, the function will return a matrix of zeros with the same shape as would be expected from a normal MI calculation, along with metadata indicating this special case:

```python
{
    'scores': zeros_matrix,
    'coupling_matrix': zeros_matrix,
    'method': 'mutual_information',
    'top_pairs': [],
    'params': {
        'pseudocount': pseudocount,
        'alphabet_size': alphabet_size,
        'single_sequence': True  # Flag to indicate this was a single-sequence case
    }
}
```

## 4. Code Implementation

### 4.1 Updates to `mutual_information.py`

```python
def calculate_mutual_information(msa_sequences, pseudocount=None, verbose=False):
    """
    Calculate mutual information between positions in the MSA.
    This is simpler and faster than DCA methods.
    
    Parameters:
    -----------
    msa_sequences : list of str
        List of aligned RNA sequences from an MSA
    pseudocount : float or None, default=None
        Pseudocount value to use for frequency normalization.
        If None, will use adaptive selection based on MSA size.
        If 0.0, no pseudocounts will be used (original behavior).
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
        - 'params': dictionary with parameters used for calculation
    """
    if not msa_sequences:
        if verbose:
            print("No sequences provided")
        return None
    
    # NEW CODE: Check for single-sequence MSA
    unique_sequences = set(msa_sequences)
    if len(unique_sequences) <= 1:
        seq_len = len(msa_sequences[0])
        if verbose:
            print(f"Single-sequence MSA detected, skipping MI calculation for sequence of length {seq_len}")
        
        # Create zero matrix
        mi_matrix = np.zeros((seq_len, seq_len))
        
        # Get adaptive pseudocount if not specified
        if pseudocount is None:
            pseudocount = get_adaptive_pseudocount(msa_sequences)
        
        # Return the same structure as expected from the full calculation
        return {
            'scores': mi_matrix,
            'coupling_matrix': mi_matrix,
            'method': 'mutual_information',
            'top_pairs': [],
            'params': {
                'pseudocount': pseudocount,
                'alphabet_size': len(['A', 'C', 'G', 'U', 'T', '-', 'N']),
                'single_sequence': True  # Flag to indicate this was a single-sequence case
            },
            'calculation_time': 0.0
        }
    
    # Get dimensions
    n_seqs = len(msa_sequences)
    seq_len = len(msa_sequences[0])
    
    # Rest of the existing function...
```

### 4.2 Updates to `enhanced_mi.py`

```python
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
    ... (existing documentation) ...
    """
    if not msa_sequences:
        return None
    
    # NEW CODE: Check for single-sequence MSA
    unique_sequences = set(msa_sequences)
    if len(unique_sequences) <= 1:
        seq_len = len(msa_sequences[0])
        if verbose:
            logger.info(f"Single-sequence MSA detected, skipping enhanced MI calculation for sequence of length {seq_len}")
        
        # Create zero matrices
        mi_matrix = np.zeros((seq_len, seq_len))
        
        # Get adaptive pseudocount if not specified
        if pseudocount is None:
            pseudocount = get_adaptive_pseudocount(msa_sequences)
        
        # Define alphabet for consistent output
        alphabet = ['A', 'C', 'G', 'U', 'T', '-', 'N']
        
        # Return result with the same structure as the full calculation
        return {
            'mi_matrix': mi_matrix,
            'apc_matrix': mi_matrix,
            'scores': mi_matrix,
            'coupling_matrix': mi_matrix,
            'method': 'mutual_information_enhanced',
            'top_pairs': [],
            'params': {
                'pseudocount': pseudocount,
                'alphabet_size': len(alphabet),
                'gap_threshold': gap_threshold,
                'conservation_range': conservation_range,
                'single_sequence': True  # Flag to indicate this was a single-sequence case
            }
        }
    
    # Get sequence count and length
    seq_count = len(msa_sequences)
    seq_length = len(msa_sequences[0])
    
    # Rest of the existing function...
```

## 5. Compatibility Considerations

### 5.1 API Compatibility

The proposed solution maintains full API compatibility with the existing implementation:
- Return values have the same structure and keys
- Calling code doesn't need to be modified
- All downstream processes that use MI features will continue to work

### 5.2 Data Format Compatibility

- The zeros matrix has the same dimensions as the expected MI matrix
- All expected keys are present in the returned dictionary
- Additional metadata indicates the special case without breaking existing consumers

## 6. Testing and Validation

### 6.1 Test Cases

The optimization should be tested with:

1. **Normal MSAs** - Ensure no changes in output for MSAs with multiple unique sequences
2. **Single-sequence MSAs** - Verify optimization is triggered and zeros matrix is returned
3. **Multiple identical sequences** - Confirm detection of duplicate sequences works
4. **Long single-sequence MSAs** - Measure performance improvement for large sequences

### 6.2 Validation Metrics

1. **Correctness** - Verify output structure remains compatible with downstream processes
2. **Performance** - Measure execution time improvement for single-sequence MSAs
3. **Memory Usage** - Confirm reduced memory footprint for long single-sequence MSAs

## 7. Expected Performance Improvement

### 7.1 Time Savings

For a single-sequence MSA:
- MI calculation time will reduce from O(n²) to O(1) where n is sequence length
- For a 3,000 nt sequence, this could save minutes of computation time

### 7.2 Memory Optimization

For large single-sequence MSAs:
- Memory requirements will be significantly reduced
- Temporary data structures for counting and frequency analysis will be avoided
- Peak memory usage should remain nearly constant regardless of sequence length