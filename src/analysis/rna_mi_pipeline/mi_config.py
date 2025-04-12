# RNA MI Pipeline Configuration and Parameters

"""
This configuration file defines the optimal parameters for the RNA MI pipeline
based on extensive validation testing.

Use these parameters as a starting point for your specific dataset or integration
with your workflow.
"""

import os

# Basic configuration with recommended parameters
DEFAULT_CONFIG = {
    # Core parameters
    'max_length': 750,       # Maximum sequence length to process without chunking
    'chunk_size': 600,       # Size of each chunk for long sequences
    'overlap': 200,          # Overlap between chunks
    'gap_threshold': 0.5,    # Maximum gap frequency for filtering
    'identity_threshold': 0.8,  # Sequence identity threshold for filtering
    'max_sequences': 5000,   # Maximum number of sequences to use from MSA
    'conservation_range': (0.2, 0.95),  # Range of conservation for position filtering
    
    # Pseudocount parameters
    'pseudocount': None,     # Pseudocount value (None for adaptive selection)
    'use_adaptive_pseudocount': True,   # Whether to adapt based on MSA size
    
    # Execution parameters
    'parallel': True,        # Enable parallel processing
    'n_jobs': None,          # Number of jobs for parallel processing (None = CPU count - 1)
    'timeout': 3600,         # Timeout in seconds for processing a single RNA
    
    # Memory optimization parameters
    'batch_size': 5000,      # Batch size for MI calculation
}

# Hardware-specific optimizations
# These parameter sets are optimized for specific hardware configurations
HARDWARE_CONFIGS = {
    'standard_workstation': {
        # For a standard workstation with 8-16 cores and 32-64GB RAM
        'n_jobs': 8,
        'batch_size': 5000,
        'max_sequences': 10000,
    },
    'limited_resources': {
        # For limited hardware (4 cores, 16GB RAM)
        'n_jobs': 3,
        'batch_size': 1000,
        'max_sequences': 3000,
        'chunk_size': 500,  # Smaller chunks to reduce peak memory
    },
    'high_performance': {
        # For high-performance systems (32+ cores, 128+ GB RAM)
        'n_jobs': 24,
        'batch_size': 10000,
        'max_sequences': 20000,
    }
}

# RNA length-specific parameters
# These parameter sets are optimized for different RNA length ranges
RNA_LENGTH_CONFIGS = {
    'short': {  # <300 nt
        'max_length': 300,
        'chunk_size': None,  # No chunking needed
        'overlap': None,
    },
    'medium': {  # 300-750 nt
        'max_length': 750,
        'chunk_size': 600,
        'overlap': 150,
    },
    'long': {  # 750-1500 nt
        'max_length': 750,
        'chunk_size': 600,
        'overlap': 200,
    },
    'very_long': {  # >1500 nt
        'max_length': 750,
        'chunk_size': 500,
        'overlap': 250,  # Increased overlap for better consistency
    }
}

# MSA quality-specific parameters
# These parameter sets are optimized for different MSA qualities
MSA_QUALITY_CONFIGS = {
    'high_quality': {  # Many diverse sequences
        'gap_threshold': 0.4,
        'identity_threshold': 0.85,
        'max_sequences': 10000,
        'conservation_range': (0.3, 0.9),
        'pseudocount': 0.2,  # Lower pseudocount for high-quality MSAs
    },
    'medium_quality': {  # Standard MSAs
        'gap_threshold': 0.5,
        'identity_threshold': 0.8,
        'max_sequences': 5000,
        'conservation_range': (0.2, 0.95),
        'pseudocount': 0.5,  # Default value
    },
    'low_quality': {  # Few or highly similar sequences
        'gap_threshold': 0.6,  # More permissive gap threshold
        'identity_threshold': 0.7,  # More aggressive sequence clustering
        'max_sequences': 1000,
        'conservation_range': (0.1, 0.99),  # Wider conservation range
        'pseudocount': 0.8,  # Higher pseudocount for low-quality MSAs
    }
}

def get_config(hardware_profile='standard_workstation', 
              rna_length='medium', 
              msa_quality='medium_quality'):
    """
    Get configuration parameters based on hardware, RNA length, and MSA quality.
    
    Parameters:
    -----------
    hardware_profile : str
        Hardware profile ('standard_workstation', 'limited_resources', 'high_performance')
    rna_length : str
        RNA length category ('short', 'medium', 'long', 'very_long')
    msa_quality : str
        MSA quality category ('high_quality', 'medium_quality', 'low_quality')
        
    Returns:
    --------
    dict
        Combined configuration parameters
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Update with hardware-specific parameters
    if hardware_profile in HARDWARE_CONFIGS:
        config.update(HARDWARE_CONFIGS[hardware_profile])
    
    # Update with RNA length-specific parameters
    if rna_length in RNA_LENGTH_CONFIGS:
        config.update(RNA_LENGTH_CONFIGS[rna_length])
    
    # Update with MSA quality-specific parameters
    if msa_quality in MSA_QUALITY_CONFIGS:
        config.update(MSA_QUALITY_CONFIGS[msa_quality])
    
    return config

def get_memory_optimized_config(available_memory_gb, sequence_length, num_sequences):
    """
    Get memory-optimized configuration based on available memory and data size.
    
    Parameters:
    -----------
    available_memory_gb : float
        Available memory in GB
    sequence_length : int
        RNA sequence length
    num_sequences : int
        Number of sequences in MSA
        
    Returns:
    --------
    dict
        Memory-optimized configuration
    """
    # Start with limited resources config
    config = get_config(hardware_profile='limited_resources')
    
    # Calculate approximate memory requirements
    # MI matrix: 8 bytes per element * sequence_length^2
    mi_matrix_mb = 8 * sequence_length**2 / (1024**2)
    
    # Sequence data: ~1 byte per character * sequence_length * num_sequences
    seq_data_mb = sequence_length * num_sequences / (1024**2)
    
    # Total with overhead factor of 3
    estimated_memory_mb = (mi_matrix_mb + seq_data_mb) * 3
    
    # Calculate available memory in MB
    available_memory_mb = available_memory_gb * 1024
    
    # Calculate safe number of sequences
    if estimated_memory_mb > available_memory_mb * 0.8:
        safe_sequences = int(num_sequences * available_memory_mb * 0.8 / estimated_memory_mb)
        config['max_sequences'] = min(config['max_sequences'], max(1000, safe_sequences))
    
    # Adjust batch size based on available memory
    memory_factor = min(1.0, available_memory_mb / 16384)  # Normalized to 16GB
    config['batch_size'] = max(1000, int(config['batch_size'] * memory_factor))
    
    # Adjust chunk parameters for long sequences
    if sequence_length > 1000:
        # Smaller chunks for very long sequences when memory is limited
        chunk_factor = min(1.0, available_memory_mb / 32768)  # Normalized to 32GB
        config['chunk_size'] = max(400, int(config['chunk_size'] * chunk_factor))
        config['overlap'] = max(150, int(config['chunk_size'] * 0.33))  # 1/3 of chunk size
    
    return config

# Example usage:
if __name__ == "__main__":
    # Print default configuration
    print("Default configuration:")
    for key, value in DEFAULT_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Get configuration for different scenarios
    print("\nConfiguration for standard workstation, medium RNA, medium MSA quality:")
    config = get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nConfiguration for limited resources, long RNA, low MSA quality:")
    config = get_config('limited_resources', 'long', 'low_quality')
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nMemory-optimized configuration for 8GB RAM, 2000nt RNA, 10000 sequences:")
    config = get_memory_optimized_config(8, 2000, 10000)
    for key, value in config.items():
        print(f"  {key}: {value}")