import os
import json
import shutil
from typing import Dict, Tuple
import lm_eval

from utils import CustomJSONEncoder

class BenchmarkModels:
    def __init__(self, args):
        self.args = args
        self.results_path = os.path.join("benchmark_results", self.args.model_name.split("/")[-1])
        self.current_subfolder = None

    def benchmark(self, model_name: str) -> Dict:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name}",
            tasks=self.args.benchmarks,
            log_samples=False,
            batch_size=self.args.batch_size,
            limit=self.args.limit,
            device=self.args.device
        )
        results['results']['model_name'] = model_name
        return results

    def get_new_subfolder(self) -> str:
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            return "00"
        subfolders = [f for f in os.listdir(self.results_path) 
                     if os.path.isdir(os.path.join(self.results_path, f))]
        numbers = [int(f) for f in subfolders if f.isdigit()]
        return f"{max(numbers) + 1:02}" if numbers else "00"

    def save_results(self, results: Dict, model_type: str):
        if self.current_subfolder is None:
            self.current_subfolder = self.get_new_subfolder()
        
        output_dir = os.path.join(self.results_path, self.current_subfolder)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_type}_eval.json")
        
        with open(output_path, "w") as fp:
            json.dump(results, fp, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

        if model_type == "base_model":
            main_output_path = os.path.join(self.results_path, "base_model_eval.json")
            shutil.copy2(output_path, main_output_path)

    def run_benchmark(self) -> Tuple[Dict, Dict]:
        base_results = self._run_base_model()
        unlearned_results = self._run_unlearned_model()
        return base_results, unlearned_results

    def _run_base_model(self) -> Dict:
        base_model_path = os.path.join(self.results_path, "base_model_eval.json")
        if os.path.exists(base_model_path) and not self.args.rerun_base:
            print("Loading existing base model evaluation...")
            with open(base_model_path, 'r') as f:
                return json.load(f)
        
        print("Running base model evaluation...")
        results = self.benchmark(self.args.model_name)
        self.save_results(results, "base_model")
        return results

    def _run_unlearned_model(self) -> Dict:
        print("Running unlearned model evaluation...")
        results = self.benchmark(self.args.unlearned_model)
        self.save_results(results, "unlearned_model")
        return results

    def load_results(self) -> Tuple[Dict, Dict]:
        if self.current_subfolder is None:
            self.current_subfolder = max([f for f in os.listdir(self.results_path) 
                                        if os.path.isdir(os.path.join(self.results_path, f)) 
                                        and f.isdigit()])
        
        base_path = os.path.join(self.results_path, "base_model_eval.json")
        unlearned_path = os.path.join(self.results_path, self.current_subfolder, 
                                     "unlearned_model_eval.json")

        with open(base_path, 'r') as f:
            base_results = json.load(f)
        with open(unlearned_path, 'r') as f:
            unlearned_results = json.load(f)

        return base_results, unlearned_results