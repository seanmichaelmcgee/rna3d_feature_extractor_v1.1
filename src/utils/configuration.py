"""
Configuration Module

This module handles configuration parameters for the RNA 3D Feature Extractor.
It provides functionality to load configuration from files, provide defaults,
and support environment-specific configuration.
"""

import os
import json
import logging
from pathlib import Path
import platform

class Configuration:
    """
    Manages configuration parameters for RNA 3D Feature Extractor.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize with optional configuration file.
        
        Args:
            config_file (str or Path, optional): Path to configuration file. Defaults to None.
        """
        self.logger = logging.getLogger("Configuration")
        self.config = {}
        
        # Load default configuration
        self._load_defaults()
        
        # Load from configuration file if provided
        if config_file:
            self.load_config(config_file)
            
        # Detect environment
        self.detect_environment()
        
    def _load_defaults(self):
        """
        Load default configuration parameters.
        """
        self.config = {
            'general': {
                'verbose': False,
                'batch_size': 5,
                'memory_limit': None,  # Auto-detect
            },
            'data': {
                'data_dir': 'data',
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'thermo_dir': 'data/processed/thermo_features',
                'mi_dir': 'data/processed/mi_features'
            },
            'features': {
                'pf_scale': 1.5,  # Partition function scaling for ViennaRNA
                'pseudocount': None,  # Auto-detect based on MSA size
            },
            'processing': {
                'max_workers': None,  # Auto-detect
                'skip_existing': True,
                'validate_results': True,
            }
        }
        
    def load_config(self, config_file):
        """
        Load configuration from file.
        
        Args:
            config_file (str or Path): Path to configuration file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            # Merge with current configuration
            self._merge_config(file_config)
            
            self.logger.info(f"Loaded configuration from {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration file: {e}")
            return False
            
    def get_config_value(self, key, default=None):
        """
        Get configuration parameter.
        
        Args:
            key (str): Configuration key (can be nested using dots, e.g., 'data.raw_dir')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Handle nested keys with dots
            if '.' in key:
                sections = key.split('.')
                current = self.config
                
                for section in sections[:-1]:
                    if section not in current:
                        return default
                    current = current[section]
                    
                return current.get(sections[-1], default)
            else:
                # Top-level key
                return self.config.get(key, default)
                
        except Exception:
            return default
            
    def set_config_value(self, key, value):
        """
        Set configuration parameter.
        
        Args:
            key (str): Configuration key (can be nested using dots, e.g., 'data.raw_dir')
            value: Value to set
            
        Returns:
            bool: True if setting was successful, False otherwise
        """
        try:
            # Handle nested keys with dots
            if '.' in key:
                sections = key.split('.')
                current = self.config
                
                for section in sections[:-1]:
                    if section not in current:
                        current[section] = {}
                    current = current[section]
                    
                current[sections[-1]] = value
            else:
                # Top-level key
                self.config[key] = value
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting configuration value: {e}")
            return False
            
    def save_config(self, config_file):
        """
        Save configuration to file.
        
        Args:
            config_file (str or Path): Path to save configuration
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            self.logger.info(f"Saved configuration to {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
            
    def detect_environment(self):
        """
        Detect and configure for environment.
        
        Returns:
            str: Detected environment type
        """
        # Detect Kaggle environment
        is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
        
        if is_kaggle:
            environment = 'kaggle'
            self.logger.info("Detected Kaggle environment")
            
            # Configure for Kaggle
            # Typically 4 cores, 16GB RAM
            self.set_config_value('general.memory_limit', 12.0)  # 12GB limit
            self.set_config_value('processing.max_workers', 4)
            
        else:
            environment = 'local'
            self.logger.info(f"Detected local environment: {platform.node()}")
            
            # Configure based on local system
            import psutil
            
            # Get system resources
            cpu_count = psutil.cpu_count(logical=False) or 1
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Set conservative limits (80% of available resources)
            self.set_config_value('general.memory_limit', memory_gb * 0.8)
            self.set_config_value('processing.max_workers', max(1, cpu_count - 1))
            
        # Set detected environment
        self.set_config_value('environment', environment)
        return environment
        
    def _merge_config(self, new_config):
        """
        Merge new configuration with existing configuration.
        
        Args:
            new_config (dict): New configuration to merge
        """
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                self._merge_config_section(self.config[key], value)
            else:
                # Replace value
                self.config[key] = value
                
    def _merge_config_section(self, target, source):
        """
        Recursively merge configuration sections.
        
        Args:
            target (dict): Target configuration section
            source (dict): Source configuration section
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                self._merge_config_section(target[key], value)
            else:
                # Replace value
                target[key] = value
                

def load_config(config_file):
    """
    Load configuration from file.
    
    Args:
        config_file (str or Path): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config = Configuration(config_file)
    return config.config
    
def get_config_value(config, key, default=None):
    """
    Get configuration parameter.
    
    Args:
        config (dict): Configuration dictionary
        key (str): Configuration key
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    if '.' in key:
        sections = key.split('.')
        current = config
        
        for section in sections[:-1]:
            if section not in current:
                return default
            current = current[section]
            
        return current.get(sections[-1], default)
    else:
        return config.get(key, default)
        
def save_config(config, config_file):
    """
    Save configuration to file.
    
    Args:
        config (dict): Configuration dictionary
        config_file (str or Path): Path to save configuration
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        return True
        
    except Exception:
        return False
        
def detect_environment():
    """
    Detect execution environment.
    
    Returns:
        dict: Environment information
    """
    # Detect Kaggle environment
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
    
    # Get system resources
    import psutil
    cpu_count = psutil.cpu_count(logical=False) or 1
    memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    if is_kaggle:
        environment = 'kaggle'
        # Kaggle typically has 4 cores, 16GB RAM
        max_workers = min(cpu_count, 4)
        memory_limit = min(memory_gb * 0.7, 12.0)  # 70% of available or max 12GB
    else:
        environment = 'local'
        # Use 80% of available resources for local execution
        max_workers = max(1, cpu_count - 1)
        memory_limit = memory_gb * 0.8
        
    return {
        'environment': environment,
        'resources': {
            'cpu_count': cpu_count,
            'memory_gb': memory_gb,
            'max_workers': max_workers,
            'memory_limit': memory_limit
        },
        'platform': {
            'system': platform.system(),
            'node': platform.node(),
            'python': platform.python_version()
        }
    }