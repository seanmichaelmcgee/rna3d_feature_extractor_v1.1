# RNA Feature Naming Standardization Guide

## Overview

This guide provides instructions for standardizing feature naming conventions across the RNA 3D Feature Extractor codebase to align with the downstream PyTorch machine learning pipeline requirements. Consistent naming is critical to ensure seamless data flow between feature extraction and model training/inference stages.

## Motivation

Our current feature extraction pipeline produces high-quality features, but some naming inconsistencies exist that could create integration challenges with the downstream PyTorch model. Standardizing these names will:

1. Eliminate the need for fallback logic in the data loader
2. Reduce potential for errors during model training
3. Improve code maintainability and readability
4. Enable more efficient debugging

## Feature Type Mapping

The following sections outline the current feature naming conventions and the standardized names required by the PyTorch pipeline.

### 1. Dihedral Features

| Current Implementation | Required Standard | Notes |
|------------------------|-------------------|-------|
| `{target_id}_dihedral_features.npz` | `{target_id}_dihedral_features.npz` | **File naming is correct** |
| `features` | `features` | **Array naming is correct** |
| `eta` | `eta` | **Array naming is correct** |
| `theta` | `theta` | **Array naming is correct** |

✅ **Current Status**: Dihedral feature naming is already well-aligned with pipeline requirements.

### 2. Thermodynamic Features

| Current Implementation | Required Standard | Notes |
|------------------------|-------------------|-------|
| `{target_id}_thermo_features.npz` | `{target_id}_thermo_features.npz` | **File naming is correct** |
| `position_entropy` | `positional_entropy` | **Rename needed** - Core vector feature |
| `base_pair_probs` | `pairing_probs` | **Rename needed** - Critical matrix feature |
| `mfe_structure` | `structure` | **Consider consolidating** to single name |
| Various scalar features | Keep as is | Already consistent |

🔄 **Current Status**: Thermodynamic features need standardization, especially `position_entropy` → `positional_entropy` and `base_pair_probs` → `pairing_probs`.

### 3. Evolutionary Features (MI)

| Current Implementation | Required Standard | Notes |
|------------------------|-------------------|-------|
| `{target_id}_features.npz` | `{target_id}_features.npz` | **File naming is correct** |
| `scores` or `mi_matrix` | `coupling_matrix` | **Rename needed** - Critical matrix feature |
| `method` | `method` | **Array naming is correct** |
| Missing in some files | `conservation` | **Add if available** - Optional vector feature |

🔄 **Current Status**: Evolutionary features need standardization, primarily `scores`/`mi_matrix` → `coupling_matrix`.

## Implementation Updates Required

### 1. Update `src/analysis/thermodynamic_analysis.py`

Current implementation:
```python
# Function that generates position_entropy
def calculate_positional_entropy(bpp_matrix):
    """Calculate Shannon entropy for each position based on base pairing probabilities."""
    # ...
    return {
        'positional_entropy': entropy,  # This name is used in return value
        'position_entropy': entropy,    # But this name is sometimes used when storing in features
        'mean_entropy': mean_entropy,
        'max_entropy': max_entropy
    }

# In extract_thermodynamic_features function
features.update(entropy_features)  # Adds both names to features dict
```

Required change:
```python
def calculate_positional_entropy(bpp_matrix):
    """Calculate Shannon entropy for each position based on base pairing probabilities."""
    # ...
    return {
        'positional_entropy': entropy,  # Use consistent name
        'mean_entropy': mean_entropy,
        'max_entropy': max_entropy
    }

# In extract_thermodynamic_features function
features.update(entropy_features)
# Ensure the correct name is included in required_features dictionary
required_features = {
    # ...
    'positional_entropy': features.get('positional_entropy', np.zeros(len(sequence))),
    # ...
}
```

### 2. Update `src/analysis/rna_mi_pipeline/enhanced_mi.py`

Current implementation:
```python
# In function chunk_and_analyze_rna or similar
result = {
    'mi_matrix': full_matrix,
    'scores': final_matrix,  # This is what's accessed downstream
    'apc_matrix': final_matrix,
    'top_pairs': top_pairs,
    'method': 'mutual_information_chunked',
    'chunks': len(chunks),
    'chunk_size': chunk_size,
    'overlap': overlap
}
```

Required change:
```python
# In function chunk_and_analyze_rna or similar
result = {
    'mi_matrix': full_matrix,
    'scores': final_matrix,
    'coupling_matrix': final_matrix,  # Add this standardized name
    'apc_matrix': final_matrix,
    'top_pairs': top_pairs,
    'method': 'mutual_information_chunked',
    'chunks': len(chunks),
    'chunk_size': chunk_size,
    'overlap': overlap
}
```

### 3. Update `src/analysis/mutual_information.py`

Current implementation:
```python
# In function calculate_mutual_information
return {
    'scores': mi_matrix,
    'method': 'mutual_information',
    'top_pairs': top_pairs,
    'calculation_time': time.time() - start_time if verbose else None
}

# In function convert_mi_to_evolutionary_features
features = {
    'coupling_matrix': mi_matrix,  # Already using the correct name here
    'method': mi_result['method'],
    'top_pairs': np.array(mi_result['top_pairs']) if mi_result['top_pairs'] else np.array([])
}
```

Required change:
```python
# In function calculate_mutual_information
return {
    'scores': mi_matrix,
    'coupling_matrix': mi_matrix,  # Add standardized name
    'method': 'mutual_information',
    'top_pairs': top_pairs,
    'calculation_time': time.time() - start_time if verbose else None
}
```

### 4. Update Data Extraction Script

If using `extract_features_simple.py` or similar scripts:

```python
# Update to ensure consistent naming
# For thermodynamic features:
features['positional_entropy'] = thermo_data.get('position_entropy', np.zeros(len(sequence)))
features['pairing_probs'] = thermo_data.get('base_pair_probs', np.zeros((len(sequence), len(sequence))))

# For MI features:
if mi_result:
    features['coupling_matrix'] = mi_result.get('scores', np.zeros((len(sequence), len(sequence))))
```

## Feature Validation Implementation

Add a validation function to ensure features meet pipeline requirements:

```python
def validate_features(features, sequence_length):
    """
    Validate features against expected formats and dimensions.
    
    Args:
        features: Dictionary of extracted features
        sequence_length: Expected sequence length for validation
        
    Returns:
        Boolean indicating if features are valid
    """
    # Check thermodynamic features
    if 'positional_entropy' not in features:
        warnings.warn("Missing required feature: positional_entropy")
        return False
        
    if 'pairing_probs' not in features:
        warnings.warn("Missing required feature: pairing_probs")
        return False
    
    # Check evolutionary features (if applicable)
    if 'coupling_matrix' in features:
        if features['coupling_matrix'].shape != (sequence_length, sequence_length):
            warnings.warn(f"coupling_matrix has incorrect shape: {features['coupling_matrix'].shape}, expected ({sequence_length}, {sequence_length})")
            return False
    
    # Check dihedral features (if applicable)
    if 'dihedral' in features and 'features' in features['dihedral']:
        if features['dihedral']['features'].shape[0] != sequence_length:
            warnings.warn(f"dihedral features has incorrect first dimension: {features['dihedral']['features'].shape}, expected first dim {sequence_length}")
            return False
            
    return True
```

## Implementation Checklist

- [x] Update thermodynamic feature naming (primary focus on `positional_entropy` and `pairing_probs`)
- [x] Update mutual information feature naming (focus on `coupling_matrix`)
- [x] Add feature validation to processing pipeline
- [ ] Update notebooks to reflect standardized naming
- [x] Add test cases to verify feature naming consistency

## Example Integration with PyTorch Pipeline

For reference, here's how these standardized features will be used in the PyTorch data loader:

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Get a sample from the dataset."""
    target_id = self.target_ids[idx]
    
    # Load features
    features = load_precomputed_features(target_id, self.features_dir)
    
    # Convert to tensors - relies on standardized names
    sample = {
        'target_id': target_id,
        'sequence_int': torch.tensor(self.sequence_to_int(self.sequences[idx]), dtype=torch.long),
        'dihedral_features': torch.tensor(features['dihedral']['features'], dtype=torch.float32),
        'pairing_probs': torch.tensor(features['thermo']['pairing_probs'], dtype=torch.float32),
        'positional_entropy': torch.tensor(features['thermo']['positional_entropy'], dtype=torch.float32),
        'coupling_matrix': torch.tensor(features['evolutionary']['coupling_matrix'], dtype=torch.float32),
        'coordinates': torch.tensor(self.coordinates[idx], dtype=torch.float32) if self.coordinates is not None else None,
        'length': len(self.sequences[idx])
    }
    
    return sample
```

## Conclusion

By implementing these standardization updates, we will ensure seamless integration between the feature extraction pipeline and the PyTorch model. The focus should be on maintaining consistency in naming conventions without changing the fundamental calculation or structure of the features themselves.
