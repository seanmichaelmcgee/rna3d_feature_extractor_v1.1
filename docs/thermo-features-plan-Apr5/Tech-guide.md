# RNA Thermodynamic Feature Extraction: Technical Guide

## Overview
This guide outlines a comprehensive set of thermodynamic features to extract from RNA sequences for the Stanford RNA 3D Folding competition. The focus is on high-value features that can be implemented quickly and reliably based on our current progress with the ViennaRNA integration.

## Core Thermodynamic Features

### Primary Features (Essential)
These features provide the fundamental structural and energetic information:

1. **Minimum Free Energy (MFE)** - Overall stability of the structure
   - Implementation: Already working via fc.mfe()
   - Format: Single float value (kcal/mol)
   - Importance: High - provides global stability information

2. **Base-Pairing Probability Matrix** - Probabilistic pairing information
   - Implementation: Working via partition function calculation (fc.pf())
   - Format: NxN matrix with values between 0-1
   - Importance: Critical - captures the ensemble nature of RNA structures
   - Validation: Should show diagonal patterns with varying probabilities

3. **Secondary Structure (Dot-Bracket Notation)** - Represents the MFE structure
   - Implementation: Already working
   - Format: String of dots, opening and closing brackets
   - Importance: High - provides skeleton of the folding pattern

4. **Ensemble Free Energy** - Represents energy of all possible structures
   - Implementation: Needs minor fix to handle list return type from fc.pf()
   - Format: Single float value (kcal/mol)
   - Importance: Medium-high - captures ensemble properties

### Secondary Features (High Value, Low Effort)
These enhance the model's understanding of RNA structure with minimal implementation cost:

5. **Per-Position Entropy** - Uncertainty in base pairing at each position
   - Implementation: Calculate from BPP matrix
   - Format: 1D array of length N 
   - Calculation: `position_entropy[i] = -sum(p * log(p) for p in bpp_matrix[i])` where p > 0

6. **Stem Features** - Identify and characterize helical regions
   - Implementation: Parse from dot-bracket structure
   - Format: List of stems with properties (start, end, length)
   - Validation: Compare with visualization of BPP matrix

7. **Loop Features** - Identify and characterize loop regions
   - Implementation: Parse from dot-bracket structure
   - Format: List of loops with properties (type, size, positions)
   - Types: Hairpin, bulge, internal, multi-branch

8. **Graph Representation** - Simple network representation of structure
   - Implementation: Create from dot-bracket notation
   - Format: Adjacency matrix or edge list
   - Features: Node degree, shortest paths between nucleotides

### Tertiary Features (If Time Permits)
These provide richer structural information but require more implementation effort:

9. **Stacking Energy** - Energy contribution from stacked pairs
   - Implementation: Calculate from nearest neighbors in stems
   - Format: 1D array or total value 

10. **Accessibility Scores** - Measure of nucleotide exposure
    - Implementation: Extract from ViennaRNA if available
    - Format: 1D array of length N

11. **Coarse-Grained Structural Elements** - Higher-level structural motifs
    - Implementation: Pattern matching in dot-bracket
    - Format: List of structural elements (junctions, kissing loops, etc.)

## Implementation Guidelines

### API Design
Implement as a pipeline of modular functions:

```python
def extract_thermodynamic_features(sequence, max_length=3000):
    """Master function to extract all thermodynamic features."""
    # Get basic thermodynamic data
    thermo_data = calculate_folding_energy_robust(sequence, max_length)
    
    # Extract standard features
    features = {}
    features.update(extract_basic_features(thermo_data, len(sequence)))
    features.update(extract_entropy_features(thermo_data, len(sequence)))
    features.update(extract_structural_elements(thermo_data, len(sequence)))
    
    # Optional advanced features if time permits
    if EXTRACT_ADVANCED_FEATURES:
        features.update(extract_advanced_features(thermo_data, len(sequence)))
    
    return features
```

### Key Integration Points

1. **ViennaRNA Integration**
   - Ensure consistent API handling across versions
   - Maintain the robust version detection logic
   - Fix the ensemble energy handling to properly extract numeric values

2. **Feature Standardization**
   - Consistent naming convention: `feature_type.feature_name`
   - Consistent array shapes and types
   - Feature metadata (min, max, units)

3. **Saving/Loading**
   - Save as NPZ with clear naming conventions
   - Include validation checksums
   - Version the feature format

### Performance Considerations

1. **Memory Management**
   - For sequences approaching 3,000 nt, the BPP matrix will be ~72MB
   - Consider sparse matrix storage for very large sequences

2. **Computation Time**
   - Most expensive operations are MFE and partition function calculation
   - Parallelize where possible for batch processing
   - Cache intermediate results

### Validation Approach

1. **Visualization-Based Validation**
   - Base-pair probability matrices should show diagonal patterns
   - Compare with known structures from PDB

2. **Statistical Validation**
   - Distribution of energy values should follow expected patterns
   - Structure statistics should match literature values

3. **Biological Validation**
   - Test on benchmark RNAs with well-understood structures
   - Compare with experimental data where available

## Feature Extraction Functions

Implement these key functions (along with appropriate helper functions):

```python
def extract_basic_features(thermo_data, seq_length):
    """Extract fundamental thermodynamic features."""
    return {
        'deltaG': float(thermo_data.get('mfe', 0.0)),
        'structure': thermo_data.get('mfe_structure', '.' * seq_length),
        'pairing_probs': get_clean_bpp_matrix(thermo_data, seq_length),
        'ensemble_energy': extract_ensemble_energy(thermo_data),
        'pairing_status': structure_to_pairing_status(thermo_data.get('mfe_structure', '.' * seq_length)),
    }

def extract_entropy_features(thermo_data, seq_length):
    """Calculate positional entropy from base-pair probabilities."""
    bpp_matrix = get_clean_bpp_matrix(thermo_data, seq_length)
    position_entropy = np.zeros(seq_length)
    
    # Calculate Shannon entropy for each position
    for i in range(seq_length):
        probs = bpp_matrix[i]
        nonzero_probs = probs[probs > 0]
        if len(nonzero_probs) > 0:
            position_entropy[i] = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    
    return {
        'position_entropy': position_entropy,
        'mean_entropy': np.mean(position_entropy),
        'max_entropy': np.max(position_entropy)
    }

def extract_structural_elements(thermo_data, seq_length):
    """Extract stems, loops, and other structural elements."""
    structure = thermo_data.get('mfe_structure', '.' * seq_length)
    
    # Find stems (paired regions)
    stems = find_stems(structure)
    
    # Find loops (unpaired regions)
    loops = find_loops(structure)
    
    # Create simple adjacency matrix from structure
    adjacency = structure_to_adjacency(structure)
    
    return {
        'stems': stems,
        'loops': loops,
        'adjacency': adjacency,
        'num_stems': len(stems),
        'num_loops': len(loops),
        'largest_stem': max([s['length'] for s in stems]) if stems else 0,
        'largest_loop': max([l['size'] for l in loops]) if loops else 0
    }

def extract_advanced_features(thermo_data, seq_length):
    """Extract more complex thermodynamic features if time permits."""
    # Implementation depends on progress and time available
    return {}
```

## Debug and Test Workflow

1. **Unit Tests**
   - Test on small RNA sequences with known structures
   - Verify against literature values
   - Test edge cases (very short, very long sequences)

2. **Visualization Tests**
   - Verify BPP matrix patterns match expected structures
   - Compare dot-bracket visualization with BPP matrix

3. **Integration Tests**
   - Test full feature extraction pipeline
   - Verify consistency with previous versions
   - Test on full dataset

---

