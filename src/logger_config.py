import logging
import sys

def setup_logger(name="wmdp_rec"):
    """Configure and return a logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added already
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('wmdp_rec.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger
