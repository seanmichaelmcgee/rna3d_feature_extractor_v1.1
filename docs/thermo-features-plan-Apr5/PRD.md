# RNA Thermodynamic Feature Extraction Module

## Product Requirements Document

**Version:** 1.0.0  
**Last Updated:** April 5, 2025  
**Status:** Draft

---

## 1. Document Overview

### 1.1 Purpose
This Product Requirements Document (PRD) outlines the requirements for developing a comprehensive RNA Thermodynamic Feature Extraction Module for the Stanford RNA 3D Folding Kaggle competition. The module will extract biologically relevant thermodynamic features from RNA sequences to support 3D structure prediction models.

### 1.2 Scope
This document covers the design, implementation, and integration of thermodynamic feature extraction capabilities, focusing on features derivable from RNA secondary structure prediction algorithms.

### 1.3 Definitions and Acronyms
- **RNA:** Ribonucleic Acid
- **MFE:** Minimum Free Energy
- **BPP:** Base Pair Probability
- **ViennaRNA:** A software package for RNA secondary structure prediction
- **NPZ:** NumPy Compressed Array File Format
- **TM-Score:** Template Modeling Score (structure comparison metric)

---

## 2. Product Overview

### 2.1 Product Description
The RNA Thermodynamic Feature Extraction Module is a Python-based component that analyzes RNA sequences to extract features describing their folding energetics, secondary structure patterns, and ensemble characteristics. These features serve as crucial inputs to the machine learning models responsible for predicting 3D structures in the competition.

### 2.2 User Personas
1. **Data Scientist:** Requires clean, normalized features for model training
2. **RNA Biologist:** Needs biologically interpretable features for validation
3. **ML Engineer:** Needs efficient, reliable feature extraction for pipeline integration
4. **Competition Participant:** Requires compliant features that improve predictive performance

### 2.3 Product Goals
1. Extract a comprehensive set of thermodynamic features from RNA sequences
2. Ensure robustness across varying RNA lengths and compositions
3. Maintain high performance within Kaggle competition constraints
4. Support the objective of achieving competitive TM-scores

---

## 3. Functional Requirements

### 3.1 Core Feature Extraction
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1.1 | Calculate Minimum Free Energy (MFE) for RNA sequences | High | Using ViennaRNA with proper error handling |
| FR-1.2 | Generate dot-bracket secondary structure representation | High | For the MFE structure |
| FR-1.3 | Calculate complete Base Pair Probability matrix | Critical | Must explicitly calculate partition function first |
| FR-1.4 | Extract ensemble free energy | High | Properly handle various return formats from ViennaRNA |
| FR-1.5 | Support sequences up to 3,000 nucleotides | High | Must handle memory usage appropriately |

### 3.2 Derived Features
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-2.1 | Calculate per-position Shannon entropy | High | Based on BPP distribution |
| FR-2.2 | Identify and characterize stem regions | High | Location, length, GC content |
| FR-2.3 | Identify and characterize loop regions | Medium | Type, size, sequence composition |
| FR-2.4 | Create graph representation of RNA structure | Medium | Adjacency matrix or edge list format |
| FR-2.5 | Calculate positional structural context metrics | Medium | Local structural environment |

### 3.3 Advanced Features
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-3.1 | Calculate stacking energies | Low | Energy contribution from consecutive base pairs |
| FR-3.2 | Generate accessibility scores | Low | Measure of nucleotide exposure |
| FR-3.3 | Identify higher-order structural motifs | Low | Junctions, kissing loops, pseudoknots |

### 3.4 Feature Management
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-4.1 | Save extracted features in NPZ format | High | With standardized naming conventions |
| FR-4.2 | Implement feature validation and verification | High | Check for reasonableness and biological plausibility |
| FR-4.3 | Support batch processing for multiple RNA sequences | Medium | For efficient dataset processing |
| FR-4.4 | Provide feature metadata (min, max, description) | Medium | For documentation and normalization |

---

## 4. Non-Functional Requirements

### 4.1 Performance
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| NFR-1.1 | Process a 100-nucleotide RNA in < 1 second | High | For interactive use |
| NFR-1.2 | Process a 1,000-nucleotide RNA in < 1 minute | High | For batch processing |
| NFR-1.3 | Memory usage < 500MB for 3,000-nucleotide sequence | High | To stay within Kaggle constraints |
| NFR-1.4 | Complete feature extraction for all competition sequences in < 2 hours | Medium | For pipeline efficiency |

### 4.2 Reliability
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| NFR-2.1 | Feature extraction success rate > 99% | High | Fallback mechanisms for failures |
| NFR-2.2 | Graceful handling of invalid sequences | High | Proper validation and error messages |
| NFR-2.3 | Consistent results across multiple runs | High | Deterministic behavior |
| NFR-2.4 | Robust error handling and reporting | Medium | Clear error messages and logging |

### 4.3 Compatibility
| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| NFR-3.1 | Support different ViennaRNA versions | High | For portability across environments |
| NFR-3.2 | Function without internet access | Critical | For Kaggle competition compliance |
| NFR-3.3 | Compatible with Python 3.8+ | High | For modern environment support |
| NFR-3.4 | No external executable dependencies | Critical | For Kaggle compatibility |

---

## 5. Technical Specifications

### 5.1 Architecture

#### 5.1.1 Component Diagram
```
RNA Sequence → [ViennaRNA Integration] → Thermodynamic Data → [Feature Extractors] → Feature Set → [Storage/Serialization] → NPZ Files
```

#### 5.1.2 Module Structure
- `thermodynamic_analysis.py`: Core functionality for ViennaRNA integration
- `feature_extraction.py`: Feature extraction from thermodynamic data
- `structural_elements.py`: Stem, loop, and motif detection
- `graph_features.py`: Network-based representation and analysis
- `batch_processor.py`: Multi-sequence processing utilities
- `visualization.py`: Feature visualization and validation

### 5.2 API Specification

#### 5.2.1 Primary Function
```python
def extract_thermodynamic_features(sequence: str, max_length: int = 3000) -> Dict[str, Any]:
    """
    Extract comprehensive thermodynamic features from RNA sequence.
    
    Args:
        sequence: RNA sequence (A, C, G, U/T)
        max_length: Maximum sequence length to process
        
    Returns:
        Dictionary of thermodynamic features
    """
```

#### 5.2.2 Feature Dictionary Schema
```
{
    'deltaG': float,                         # MFE in kcal/mol
    'structure': str,                        # Dot-bracket notation
    'pairing_probs': np.ndarray,             # NxN BPP matrix
    'ensemble_energy': float,                # Ensemble free energy
    'position_entropy': np.ndarray,          # 1D entropy array
    'stems': List[Dict],                     # Stem properties
    'loops': List[Dict],                     # Loop properties
    'adjacency': np.ndarray,                 # Structure graph
    'structural_metrics': Dict[str, float],  # Summary statistics
    ...
}
```

#### 5.2.3 Feature NPZ Format
```
filename: {target_id}_thermodynamic_features.npz
contents:
    - deltaG: float
    - structure: str (as bytes)
    - pairing_probs: np.ndarray (NxN)
    - position_entropy: np.ndarray (N)
    - [additional features]
```

### 5.3 Dependencies

#### 5.3.1 External Libraries
- NumPy: Array operations and file storage
- ViennaRNA: RNA secondary structure prediction (if available)
- Matplotlib: Optional for visualization

#### 5.3.2 Input Requirements
- Valid RNA sequences (A, C, G, U/T characters)
- Properly formatted target IDs

---

## 6. Data Requirements

### 6.1 Input Data
- RNA sequences from competition CSV files
- Sequence length range: 10-3,000 nucleotides
- Characters: A, C, G, U (T will be converted to U)
- Support for non-canonical nucleotides (convert to N)

### 6.2 Output Data
- NPZ files containing feature dictionaries
- Feature validation metrics and statistics
- Visualization outputs for debugging (optional)

### 6.3 Data Validation
- Check for valid RNA characters
- Validate structure balance (matching brackets)
- Verify reasonable energy ranges
- Check probability matrix constraints (values in [0,1])

---

## 7. User Experience

### 7.1 Interface
- Command-line interface for batch processing
- Python API for integration with notebooks
- Optional visualization for feature inspection

### 7.2 Error Handling
- Clear error messages with suggested solutions
- Graceful fallbacks for missing dependencies
- Detailed logging for debugging
- Progress reporting for long-running operations

---

## 8. Constraints and Limitations

### 8.1 Competition Constraints
- No internet access during Kaggle notebook execution
- Limited CPU and memory resources
- Maximum runtime of 8 hours for submission notebooks
- No external executable dependencies

### 8.2 Technical Limitations
- ViennaRNA may not be available in all environments
- Memory scaling with sequence length (O(n²) for BPP matrix)
- Computation time scaling (O(n³) for partition function)
- No support for pseudoknot structures in ViennaRNA

---

## 9. Implementation Plan

### 9.1 Phase 1: Core Implementation (Days 1-3)
- Setup development environment
- Implement basic ViennaRNA integration with robust error handling
- Fix ensemble energy handling
- Implement core feature extraction (MFE, structure, BPP matrix)
- Add position entropy calculation
- Create basic test suite

### 9.2 Phase 2: Secondary Features (Days 4-5)
- Implement stem and loop detection
- Add graph representation
- Create batch processing capability
- Implement NPZ serialization
- Add visualization tools for validation

### 9.3 Phase 3: Integration and Optimization (Days 6-7)
- Integrate with ML pipeline
- Optimize memory usage for large sequences
- Add advanced features if time permits
- Complete documentation
- Perform final testing and validation

---

## 10. Success Metrics

### 10.1 Technical Metrics
- Feature extraction success rate > 99%
- Performance within specified time limits
- Memory usage within constraints
- Test coverage > 80%

### 10.2 Functional Metrics
- Correlation of features with known RNA structures
- Feature importance in ML models
- Improvement in TM-score predictions
- Biological plausibility of extracted features

---

## 11. Approval and Sign-off

| Role | Name | Approval Date |
|------|------|---------------|
| Project Lead | _____________ | _____________ |
| Technical Lead | _____________ | _____________ |
| Data Scientist | _____________ | _____________ |
| QA Engineer | _____________ | _____________ |

---

## 12. Appendix

### 12.1 References
- ViennaRNA Package documentation
- Stanford RNA 3D Folding competition guidelines
- RNA thermodynamics literature
- Previous thermodynamic analysis implementations

### 12.2 Glossary
- **RNA Secondary Structure**: The 2D representation of RNA base pairing patterns
- **Minimum Free Energy (MFE)**: The lowest free energy state of an RNA structure
- **Base Pair Probability (BPP)**: Probability of two bases being paired in the ensemble
- **Partition Function**: Sum over all possible secondary structures, weighted by their Boltzmann factors
- **Shannon Entropy**: Measure of uncertainty in base pairing

### 12.3 Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-04-03 | | Initial draft |
| 1.0 | 2025-04-05 | | Final version |
