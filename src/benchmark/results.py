import os
import json
import shutil
import sys
import logging

from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any
from utils import CustomJSONEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exceptions import BenchmarkError, ConfigurationError

logger = logging.getLogger(__name__)

class ResultsManager:
    """
    Manages benchmark results storage and retrieval.

    This class handles:
    - Creating and managing result directories
    - Saving benchmark results to JSON files
    - Loading existing results
    - Managing result versioning through subfolders

    Attributes:
        results_path (Path): Base path for storing results
        current_subfolder (Optional[str]): Current active subfolder for results
    """

    def __init__(self, results_path: Union[str, Path]) -> None:
        """
        Initialize the results manager.

        Args:
            results_path: Base directory path for storing results
        """
        self.results_path = Path(results_path)
        self.current_subfolder: Optional[str] = None

    def get_new_subfolder(self) -> str:
        """
        Get the next available subfolder number for storing results.

        Returns:
            str: Two-digit string representing the next available folder number (e.g. '00', '01')

        Note:
            - Creates the results directory if it doesn't exist
            - Scans existing numbered folders and increments the highest number found
            - Returns '00' if no numbered folders exist
            - Maintains consistent two-digit format (e.g., '00', '01', ..., '99')

        Example:
            If folders '00' and '01' exist, returns '02'
            If no folders exist, returns '00'
        """
        if not self.results_path.exists():
            self.results_path.mkdir(parents=True)
            return "00"
        
        subfolders = [f for f in self.results_path.iterdir() if f.is_dir()]
        numbers = [int(f.name) for f in subfolders if f.name.isdigit()]
        return f"{max(numbers) + 1:02}" if numbers else "00"

    def get_base_model_path(self) -> Path:
        """Get the path for base model results."""
        return os.path.join(self.results_path, "base_model_eval.json")

    def save_results(self, results: Dict, model_type: str) -> None:
        """Save benchmark results to JSON file."""
        try:
            if not isinstance(results, dict):
                raise ValueError("Results must be a dictionary")
                
            if not model_type:
                raise ValueError("model_type cannot be empty")
                
            if self.current_subfolder is None:
                self.current_subfolder = self.get_new_subfolder()
                
            output_dir = Path(self.results_path) / self.current_subfolder
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{model_type}_eval.json"
            
            with open(output_path, "w", encoding='utf-8') as fp:
                try:
                    json.dump(results, fp, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                except TypeError as e:
                    raise ValueError(f"Results contain non-serializable data: {e}")
                    
            logger.info(f"Benchmark results saved to {output_path}")
            
            if model_type == "base_model":
                main_output_path = Path(self.results_path) / "base_model_eval.json"
                shutil.copy2(output_path, main_output_path)
                logger.info(f"Base model results also saved/updated at {main_output_path}")
                
        except Exception as e:
            raise BenchmarkError(f"Failed to save results: {str(e)}") from e

    def load_results(self) -> Tuple[Dict, Dict]:
        """Load both base and unlearned results."""
        if self.current_subfolder is None:
            subfolders = [f for f in self.results_path.iterdir() if f.is_dir() and f.name.isdigit()]
            self.current_subfolder = max(f.name for f in subfolders)
        
        base_path = self.get_base_model_path()
        unlearned_path = os.path.join(self.results_path, self.current_subfolder, "unlearned_model_eval.json")

        return (
            self.load_json(base_path),
            self.load_json(unlearned_path)
        )

    def save_json(self, data: Dict, path: Union[str, Path]) -> None:
        """
        Save data to a JSON file with proper encoding.

        Args:
            data: Dictionary containing data to save
            path: Path where the JSON file should be saved
        """
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=CustomJSONEncoder)

    def load_json(self, path: Union[str, Path]) -> Dict:
        """
        Load data from a JSON file.

        Args:
            path: Path to the JSON file to load

        Returns:
            Dictionary containing the loaded data

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """
        Validate the structure and content of benchmark results.

        Args:
            results: Dictionary containing benchmark results

        Returns:
            bool: True if results are valid, False otherwise

        Example:
            >>> results = {'results': {'wmdp': {'acc,none': 0.75}}}
            >>> validate_results(results)
            True
        """
        required_keys = ['results']
        if not all(key in results for key in required_keys):
            return False
        
        if not isinstance(results['results'], dict):
            return False
            
        return True

    def format_results_for_logging(
        self,
        base_results: Dict[str, Any],
        unlearned_results: Dict[str, Any]
    ) -> str:
        """
        Format benchmark results for logging output.

        Args:
            base_results: Results from base model evaluation
            unlearned_results: Results from unlearned model evaluation

        Returns:
            Formatted string containing comparison results

        Example:
            >>> print(format_results_for_logging(base_results, unlearned_results))
            Base Model WMDP: 75.0%
            Unlearned Model WMDP: 73.5%
            Performance Change: -1.5%
        """