"""
Utility functions for the benchmarking system.

This module contains shared utility functions used across the codebase including:
- YAML configuration loading
- Custom JSON encoding for special types
- Common data type conversions
"""

from typing import Any, Dict, Union, Optional, Tuple
from pathlib import Path
import yaml
import argparse
import json
import numpy as np
import torch
from transformers import PreTrainedModel
import os
import hashlib
import logging
from datetime import datetime

def load_yaml_config(file_path: Union[str, Path]) -> argparse.Namespace:
    """
    Load and parse a YAML configuration file into an argparse.Namespace object.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Namespace object containing configuration parameters

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed

    Example:
        >>> config = load_yaml_config('yaml_files/benchmarking/zephyr-7b.yaml')
        >>> print(config.model_name)
        'HuggingFaceH4/zephyr-7b-beta'
        >>> print(config.batch_size)
        5
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    args = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(args, sub_key, sub_value)
        else:
            setattr(args, key, value)
    
    return args

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling special PyTorch and NumPy types.

    This encoder handles:
    - NumPy integers and floats
    - NumPy arrays
    - PyTorch tensors
    - PyTorch dtypes
    - Callable objects
    """

    def default(self, obj: Any) -> Any:
        """
        Convert special types to JSON-serializable formats.

        Args:
            obj: Object to be serialized

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
        if callable(obj):
            return obj.__name__
        return super(CustomJSONEncoder, self).default(obj)

def forward_with_cache(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    module: torch.nn.Module,
    no_grad: bool = False
) -> torch.Tensor:
    """
    Perform a forward pass while caching intermediate activations.

    Args:
        model: The model to run forward pass on
        inputs: Dictionary of input tensors (e.g., 'input_ids', 'attention_mask')
        module: Specific module to cache activations from
        no_grad: Whether to run in no_grad mode

    Returns:
        Tensor of cached activations from the specified module

    Raises:
        RuntimeError: If forward pass fails
        AttributeError: If module cannot be found in model

    Note:
        This function temporarily modifies the module's forward hook and restores
        it after execution.
    """

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration parameters."""
    required_fields = [
        'model_name', 'num_epochs', 'num_batches', 'batch_size',
        'learning_rate', 'steering_coefficient', 'alpha',
        'forget_layer_id', 'optimizer_param_layer_id', 'update_layer_ids'
    ]
    
    for field in required_fields:
        if field not in config.get('args', {}):
            raise ConfigurationError(f"Missing required configuration field: {field}")
    
    # Validate numeric ranges
    args = config['args']
    if args['num_epochs'] < 1:
        raise ConfigurationError("num_epochs must be >= 1")
    if args['batch_size'] < 1:
        raise ConfigurationError("batch_size must be >= 1")
    if args['learning_rate'] <= 0:
        raise ConfigurationError("learning_rate must be > 0")
    if args['alpha'] < 0:
        raise ConfigurationError("alpha must be >= 0")

def validate_paths(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that all paths in the configuration exist or are creatable.

    Args:
        config: Configuration dictionary containing paths

    Returns:
        Tuple containing:
        - bool: Whether all paths are valid
        - Optional[str]: Error message if invalid, None if valid
    """

class ModelCache:
    """
    Manages caching of model evaluation results to avoid redundant computation.

    This class handles:
    - Caching of model evaluation results
    - Cache invalidation based on model and configuration changes
    - Disk space management for cache storage

    Attributes:
        cache_dir (Path): Directory for storing cache files
        max_cache_size (int): Maximum cache size in bytes
        current_cache_size (int): Current cache size in bytes
    """

    def __init__(self, cache_dir: str, max_cache_size: int = 1024 * 1024 * 1024) -> None:
        """
        Initialize the model cache.

        Args:
            cache_dir: Directory to store cache files
            max_cache_size: Maximum cache size in bytes (default: 1GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.current_cache_size = self._calculate_cache_size()

    def get_cache_key(self, model_name: str, config: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a model and configuration.

        Args:
            model_name: Name or path of the model
            config: Configuration dictionary

        Returns:
            String hash representing the cache key
        """
        config_str = json.dumps(config, sort_keys=True)
        combined = f"{model_name}_{config_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_cached_results(
        self, 
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached results if they exist.

        Args:
            cache_key: Cache key to look up

        Returns:
            Cached results dictionary if found, None otherwise
        """

def setup_benchmark_logging(
    log_dir: str,
    model_name: str,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging for benchmark runs with proper formatting and file handling.

    Args:
        log_dir: Directory to store log files
        model_name: Name of the model being benchmarked
        log_level: Logging level to use

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_benchmark_logging("logs", "gpt2")
        >>> logger.info("Starting benchmark run")
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name based on model and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")

    # Configure logger
    logger = logging.getLogger(f"benchmark_{model_name}")
    logger.setLevel(log_level)

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    return logger

def get_project_root() -> Path:
    """
    Get absolute path to project root directory.
    This allows imports to work regardless of where scripts are run from.
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent