"""
Memory Monitoring Utilities for RNA Feature Extraction

This module provides utilities for tracking memory usage during RNA feature extraction
to identify bottlenecks and optimize performance for Kaggle submissions.

Usage:
    from src.analysis.memory_monitor import MemoryTracker, log_memory_usage

    # Track memory usage at key points
    log_memory_usage("Before MI calculation")
    
    # Or use the context manager for tracking sections of code
    with MemoryTracker("MI calculation"):
        # Code to calculate MI features
        mi_features = calculate_mutual_information(msa_sequences)
"""

import os
import time
import psutil
import functools
import traceback
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Global memory usage history for tracking across multiple operations
memory_history = {
    'timestamps': [],
    'usage_gb': [],
    'labels': []
}

def log_memory_usage(label="Current memory"):
    """
    Log current memory usage with an optional label.
    
    Args:
        label: Description of the memory checkpoint
        
    Returns:
        float: Current memory usage in GB
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
        
        # Record in history
        memory_history['timestamps'].append(time.time())
        memory_history['usage_gb'].append(memory_gb)
        memory_history['labels'].append(label)
        
        # Print current usage
        print(f"{label}: {memory_gb:.2f} GB")
        return memory_gb
    except Exception as e:
        print(f"Error logging memory usage: {e}")
        return 0.0

@contextmanager
def MemoryTracker(section_name="Code section"):
    """
    Context manager to track memory usage before and after a section of code.
    
    Args:
        section_name: Name of the code section being tracked
        
    Example:
        with MemoryTracker("Feature extraction"):
            features = extract_features(sequence)
    """
    start_time = time.time()
    start_mem = log_memory_usage(f"Starting {section_name}")
    
    try:
        yield
    finally:
        end_mem = log_memory_usage(f"Finished {section_name}")
        elapsed = time.time() - start_time
        memory_diff = end_mem - start_mem
        
        print(f"{section_name} completed in {elapsed:.2f} seconds")
        print(f"Memory change: {memory_diff:.2f} GB ({'+' if memory_diff >= 0 else ''}{memory_diff:.2f} GB)")

def memory_usage_decorator(func):
    """
    Decorator to track memory usage before and after a function call.
    
    Args:
        func: Function to track
        
    Returns:
        Wrapped function with memory tracking
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryTracker(func.__name__):
            return func(*args, **kwargs)
    return wrapper

def plot_memory_usage(output_file=None, show=True, clear_history=False):
    """
    Plot memory usage history from all tracking points.
    
    Args:
        output_file: Path to save the plot
        show: Whether to display the plot
        clear_history: Whether to clear the history after plotting
    """
    if not memory_history['timestamps']:
        print("No memory history to plot")
        return
    
    # Convert timestamps to relative times
    start_time = memory_history['timestamps'][0]
    rel_times = [(t - start_time) for t in memory_history['timestamps']]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(rel_times, memory_history['usage_gb'], marker='o', linestyle='-')
    
    # Add peak memory annotation
    peak_idx = np.argmax(memory_history['usage_gb'])
    peak_memory = memory_history['usage_gb'][peak_idx]
    peak_time = rel_times[peak_idx]
    plt.annotate(f'Peak: {peak_memory:.2f} GB',
                xy=(peak_time, peak_memory),
                xytext=(peak_time, peak_memory + 0.1),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    # Add labels for significant points
    for i, (t, mem, label) in enumerate(zip(rel_times, memory_history['usage_gb'], memory_history['labels'])):
        # Only label significant points to avoid clutter
        if i == 0 or i == len(rel_times)-1 or i == peak_idx or \
           (i > 0 and abs(memory_history['usage_gb'][i] - memory_history['usage_gb'][i-1]) > 0.1):
            plt.annotate(label, 
                         xy=(t, mem),
                         xytext=(t, mem - 0.2),
                         rotation=45,
                         fontsize=8,
                         ha='right')
    
    # Set plot labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage During Feature Extraction')
    plt.grid(alpha=0.3)
    
    # Add summary statistics
    plt.figtext(0.5, 0.01, 
                f"Peak memory: {peak_memory:.2f} GB | Total running time: {rel_times[-1]:.2f} seconds",
                ha='center',
                fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.1))
    
    # Save the plot if requested
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Memory usage plot saved to {output_file}")
    
    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary statistics
    print(f"Memory Usage Summary:")
    print(f"- Initial memory: {memory_history['usage_gb'][0]:.2f} GB")
    print(f"- Peak memory: {peak_memory:.2f} GB")
    print(f"- Final memory: {memory_history['usage_gb'][-1]:.2f} GB")
    print(f"- Total running time: {rel_times[-1]:.2f} seconds")
    
    # Clear history if requested
    if clear_history:
        memory_history['timestamps'] = []
        memory_history['usage_gb'] = []
        memory_history['labels'] = []
    
    return peak_memory

def profile_rna_length_memory(seq_lengths=[100, 500, 1000, 2000, 3000], output_dir=None):
    """
    Profile memory usage for different RNA sequence lengths.
    
    Args:
        seq_lengths: List of sequence lengths to test
        output_dir: Directory to save results
        
    Returns:
        dict: Memory usage results for each sequence length
    """
    # Import here to avoid circular imports
    from ..analysis import thermodynamic_analysis
    
    print(f"Profiling memory usage for RNA sequences of various lengths")
    results = {}
    
    # Create fake RNA sequences of specified lengths (using a repeating pattern)
    sequences = {}
    for length in seq_lengths:
        # Create a sequence with an approximately balanced base composition
        pattern = 'GAUCGAUCGAUAGCUAGCUAUGCAUGCAU'  # 10 repeats of this is about 300 nt
        repeats = (length // len(pattern)) + 1
        sequences[length] = (pattern * repeats)[:length]
    
    # Process each sequence and measure memory usage
    for length, sequence in sequences.items():
        print(f"\nProcessing sequence of length {length}")
        
        # Reset memory history for this sequence
        memory_history['timestamps'] = []
        memory_history['usage_gb'] = []
        memory_history['labels'] = []
        
        # Log initial memory
        log_memory_usage(f"Initial memory (length={length})")
        
        # Extract features with memory tracking
        with MemoryTracker(f"Thermodynamic features (length={length})"):
            features = thermodynamic_analysis.extract_thermodynamic_features(sequence)
        
        # Plot memory usage
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"memory_profile_length_{length}.png"
            peak_memory = plot_memory_usage(output_file=output_file, show=False)
        else:
            peak_memory = plot_memory_usage(show=False)
        
        # Store results
        results[length] = {
            'sequence_length': length,
            'peak_memory_gb': peak_memory,
            'feature_count': len(features) if features else 0,
            'processing_time': memory_history['timestamps'][-1] - memory_history['timestamps'][0]
        }
    
    # Create summary plot
    lengths = sorted(results.keys())
    peak_memory = [results[l]['peak_memory_gb'] for l in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, peak_memory, marker='o', linestyle='-')
    
    # Fit a polynomial to predict memory usage
    if len(lengths) >= 3:
        coeffs = np.polyfit(lengths, peak_memory, 2)
        poly = np.poly1d(coeffs)
        
        # Plot the fitted curve
        x_range = np.linspace(min(lengths), max(lengths)*1.2, 100)
        plt.plot(x_range, poly(x_range), 'r--', label=f'Fitted curve: {poly}')
        plt.legend()
        
        # Predict memory for longest Kaggle sequence (3000 nt)
        predicted_3k = poly(3000)
        plt.annotate(f'Predicted for 3000 nt: {predicted_3k:.2f} GB',
                     xy=(3000, predicted_3k),
                     xytext=(0.7*max(lengths), 0.8*max(peak_memory)),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('Sequence Length (nt)')
    plt.ylabel('Peak Memory Usage (GB)')
    plt.title('Memory Scaling with RNA Sequence Length')
    plt.grid(alpha=0.3)
    
    # Save summary plot
    if output_dir:
        summary_file = Path(output_dir) / "memory_scaling_summary.png"
        plt.savefig(summary_file, dpi=100, bbox_inches='tight')
        print(f"Memory scaling plot saved to {summary_file}")
    
    plt.show()
    
    # Print summary table
    print("\nMemory Scaling Summary:")
    print("----------------------")
    print("Length (nt) | Peak Memory (GB) | Processing Time (s)")
    print("-----------|-----------------|-------------------")
    for length in sorted(results.keys()):
        print(f"{length:11d} | {results[length]['peak_memory_gb']:15.2f} | {results[length]['processing_time']:19.2f}")
    
    return results

# Example usage for testing
if __name__ == "__main__":
    print("Memory Monitoring Module")
    
    # Test basic memory logging
    log_memory_usage("Initial memory")
    
    # Test context manager
    with MemoryTracker("Test allocation"):
        # Allocate some memory to see the effect
        data = [0] * 10**7
        time.sleep(1)
    
    # Test function decorator
    @memory_usage_decorator
    def test_function():
        # Allocate more memory
        data = [0] * 10**7
        time.sleep(1)
        return "Test complete"
    
    test_function()
    
    # Plot memory usage
    plot_memory_usage()
    
    print("Memory monitoring tests complete")