import os
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple
from utils import CustomJSONEncoder

class ResultsManager:
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.current_subfolder = None

    def get_new_subfolder(self) -> str:
        """Get the next available numbered subfolder."""
        if not self.results_path.exists():
            self.results_path.mkdir(parents=True)
            return "00"
        
        subfolders = [f for f in self.results_path.iterdir() if f.is_dir()]
        numbers = [int(f.name) for f in subfolders if f.name.isdigit()]
        return f"{max(numbers) + 1:02}" if numbers else "00"

    def get_base_model_path(self) -> Path:
        """Get the path for base model results."""
        return os.path.join(self.results_path, "base_model_eval.json")

    def save_results(self, results: Dict, model_type: str):
        """Save benchmark results to JSON file."""
        if self.current_subfolder is None:
            self.current_subfolder = self.get_new_subfolder()
        
        output_dir = os.path.join(self.results_path, self.current_subfolder)
        output_dir.mkdir(exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_type}_eval.json")
        
        self.save_json(results, output_path)
        print(f"Benchmark results saved to {output_path}")

        if model_type == "base_model":
            main_output_path = self.get_base_model_path()
            shutil.copy2(output_path, main_output_path)
            print(f"Base model results also saved/updated at {main_output_path}")

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

    @staticmethod
    def save_json(data: Dict, path: Path):
        """Save dictionary to JSON file."""
        with open(path, "w") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    @staticmethod
    def load_json(path: Path) -> Dict:
        """Load dictionary from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)