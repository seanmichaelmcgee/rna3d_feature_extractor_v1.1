"""
MemoryMonitor Module

This module tracks and manages memory usage during feature extraction.
It provides functionality for memory tracking, visualization, and enforcement of limits.
"""

import time
import os
import gc
import psutil
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime
import threading

class MemoryMonitor:
    """
    Monitors and manages memory usage during feature extraction.
    """
    
    def __init__(self, memory_limit=None, verbose=False, output_dir=None):
        """
        Initialize with optional memory limit.
        
        Args:
            memory_limit (float, optional): Memory limit in GB. Defaults to None.
            verbose (bool, optional): Whether to print detailed progress. Defaults to False.
            output_dir (str or Path, optional): Directory for memory plots. Defaults to None.
        """
        self.memory_limit = memory_limit
        self.verbose = verbose
        self.logger = logging.getLogger("MemoryMonitor")
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path("data/processed/memory_plots")
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize memory tracking
        self.memory_history = []
        self.tracking_active = False
        self.current_label = None
        self.tracking_thread = None
        self.tracking_interval = 0.5  # seconds
        
        if self.verbose:
            self.logger.info(f"Memory monitor initialized")
            if memory_limit:
                self.logger.info(f"Memory limit set to {memory_limit} GB")
                
    def start_tracking(self, label):
        """
        Start tracking memory usage for a specific operation.
        
        Args:
            label (str): Label for the operation being tracked
        """
        if self.tracking_active:
            self.stop_tracking()
            
        self.current_label = label
        self.tracking_active = True
        
        # Reset memory history
        self.memory_history = []
        
        # Record initial memory usage
        self._record_memory_usage(label)
        
        # Start background tracking thread
        self.tracking_thread = threading.Thread(target=self._track_memory_background)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        if self.verbose:
            self.logger.info(f"Starting {label}: {self._get_current_memory_gb():.2f} GB")
            
    def stop_tracking(self):
        """
        Stop tracking and return memory usage statistics.
        
        Returns:
            dict: Memory usage statistics
        """
        if not self.tracking_active:
            return {}
            
        # Stop tracking thread
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
            self.tracking_thread = None
            
        # Record final memory usage
        self._record_memory_usage(f"{self.current_label} (final)")
        
        # Calculate statistics
        gb_values = [entry['memory_gb'] for entry in self.memory_history]
        
        stats = {
            'label': self.current_label,
            'start_memory_gb': gb_values[0] if gb_values else 0,
            'end_memory_gb': gb_values[-1] if gb_values else 0,
            'peak_memory_gb': max(gb_values) if gb_values else 0,
            'min_memory_gb': min(gb_values) if gb_values else 0,
            'duration_seconds': self.memory_history[-1]['elapsed'] if self.memory_history else 0
        }
        
        # Calculate memory change
        stats['memory_change_gb'] = stats['end_memory_gb'] - stats['start_memory_gb']
        
        if self.verbose:
            self.logger.info(f"Finished {self.current_label}: {stats['end_memory_gb']:.2f} GB")
            self.logger.info(f"{self.current_label} completed in {stats['duration_seconds']:.2f} seconds")
            self.logger.info(f"Memory change: {stats['memory_change_gb']:.2f} GB")
            
        self.current_label = None
        return stats
        
    def log_memory_usage(self, label):
        """
        Log current memory usage with a descriptive label.
        
        Args:
            label (str): Description of the current operation
        """
        memory_gb = self._get_current_memory_gb()
        
        if self.verbose:
            self.logger.info(f"{label}: {memory_gb:.2f} GB")
            
        return memory_gb
        
    def plot_memory_usage(self, output_file=None):
        """
        Generate plot of memory usage over time.
        
        Args:
            output_file (str or Path, optional): Path to save the plot. Defaults to None.
            
        Returns:
            str: Path to saved plot or None if plotting failed
        """
        if not self.memory_history:
            self.logger.warning("No memory data to plot")
            return None
            
        try:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Extract data
            times = [entry['elapsed'] for entry in self.memory_history]
            memory_values = [entry['memory_gb'] for entry in self.memory_history]
            labels = [entry['label'] for entry in self.memory_history]
            
            # Create the plot
            plt.plot(times, memory_values, 'b-')
            
            # Add labels at major changes
            last_label = None
            for i, (t, m, label) in enumerate(zip(times, memory_values, labels)):
                if label != last_label:
                    plt.annotate(label, xy=(t, m), xytext=(5, 5), 
                                textcoords='offset points', fontsize=8)
                    last_label = label
                    
            # Add memory limit line if specified
            if self.memory_limit:
                plt.axhline(y=self.memory_limit, color='r', linestyle='--', 
                           label=f"Memory Limit ({self.memory_limit} GB)")
                
            # Set labels
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory Usage (GB)')
            plt.title('Memory Usage Over Time')
            plt.grid(True, alpha=0.3)
            
            # Determine output file path if not provided
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = self.output_dir / f"memory_usage_{timestamp}.png"
                
            # Save figure
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            if self.verbose:
                self.logger.info(f"Memory plot saved to {output_file}")
                
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error generating memory plot: {e}")
            return None
            
    def check_memory_limit(self):
        """
        Check if current memory usage is approaching the limit.
        
        Returns:
            tuple: (current_memory_gb, is_approaching_limit)
        """
        if not self.memory_limit:
            return (self._get_current_memory_gb(), False)
            
        current_memory = self._get_current_memory_gb()
        approaching_limit = current_memory > (self.memory_limit * 0.9)
        
        if approaching_limit and self.verbose:
            self.logger.warning(f"Approaching memory limit: {current_memory:.2f} GB / {self.memory_limit} GB")
            
        return (current_memory, approaching_limit)
        
    def cleanup(self):
        """
        Force garbage collection to free memory.
        
        Returns:
            float: Amount of memory freed in GB
        """
        before = self._get_current_memory_gb()
        
        # Force garbage collection
        gc.collect()
        
        after = self._get_current_memory_gb()
        freed = before - after
        
        if self.verbose and freed > 0.1:
            self.logger.info(f"Memory cleanup freed {freed:.2f} GB")
            
        return freed
        
    def _get_current_memory_gb(self):
        """
        Get current memory usage in GB.
        
        Returns:
            float: Current memory usage in GB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
        
    def _record_memory_usage(self, label=None):
        """
        Record current memory usage with timestamp.
        
        Args:
            label (str, optional): Label for the current operation. Defaults to None.
        """
        if not label:
            label = self.current_label or "Unknown"
            
        # Calculate elapsed time
        start_time = self.memory_history[0]['timestamp'] if self.memory_history else time.time()
        elapsed = time.time() - start_time
        
        # Record memory usage
        self.memory_history.append({
            'timestamp': time.time(),
            'elapsed': elapsed,
            'memory_gb': self._get_current_memory_gb(),
            'label': label
        })
        
    def _track_memory_background(self):
        """
        Background thread function to track memory usage at regular intervals.
        """
        while self.tracking_active:
            self._record_memory_usage()
            time.sleep(self.tracking_interval)


class MemoryTracker:
    """
    Context manager for tracking memory usage during a specific code block.
    This is a simplified interface to MemoryMonitor for use in with statements.
    """
    
    def __init__(self, label, memory_monitor=None, verbose=False):
        """
        Initialize the context manager.
        
        Args:
            label (str): Label for the operation being tracked
            memory_monitor (MemoryMonitor, optional): Memory monitor instance. Defaults to None.
            verbose (bool, optional): Whether to print detailed progress. Defaults to False.
        """
        self.label = label
        self.verbose = verbose
        
        # Use provided monitor or create a new one
        if memory_monitor:
            self.memory_monitor = memory_monitor
            self.owns_monitor = False
        else:
            self.memory_monitor = MemoryMonitor(verbose=verbose)
            self.owns_monitor = True
            
    def __enter__(self):
        """Start tracking when entering the context."""
        self.memory_monitor.start_tracking(self.label)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking when exiting the context."""
        self.stats = self.memory_monitor.stop_tracking()
        
        # If we created the monitor, generate a plot
        if self.owns_monitor and self.verbose:
            self.memory_monitor.plot_memory_usage()


def log_memory_usage(label):
    """
    Standalone function to log current memory usage.
    
    Args:
        label (str): Description of the current operation
        
    Returns:
        float: Current memory usage in GB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
    
    logger = logging.getLogger("MemoryMonitor")
    logger.info(f"{label}: {memory_gb:.2f} GB")
    
    return memory_gb