import logging
from typing import Optional
from pathlib import Path

def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = "optimization.log"
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Optional name for the logger (defaults to root logger)
        level: Logging level (default: INFO)
        log_file: Optional path to log file (default: "optimization.log")

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("optimization", logging.DEBUG, "custom.log")
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
