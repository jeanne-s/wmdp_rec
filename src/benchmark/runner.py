from typing import Dict, Tuple, Any
from pathlib import Path
import lm_eval
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exceptions import BenchmarkError, ConfigurationError

class BenchmarkRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the benchmark runner.

        Args:
            args: Configuration arguments including model paths and benchmark settings
        """
        self.args = args
        self.results_path = os.path.join("benchmark_results", self.args.model_name.split("/")[-1])

    def benchmark(self, model_name: str) -> Dict:
        """
        Run benchmark evaluation for a single model.
        
        Args:
            model_name: Name or path of the model to evaluate
            
        Returns:
            Dict containing evaluation results with the following structure:
            {
                'results': {
                    'model_name': str,
                    'task_name': {
                        'acc,none': float,
                        ...
                    },
                    ...
                }
            }
        """
        try:
            if not isinstance(self.args.batch_size, int) or self.args.batch_size < 1:
                raise ConfigurationError("batch_size must be a positive integer")
                
            if not self.args.benchmarks:
                raise ConfigurationError("No benchmarks specified in configuration")

            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_name},trust_remote_code={self.args.trust_remote_code}",
                tasks=self.args.benchmarks,
                log_samples=False,
                batch_size=self.args.batch_size,
                limit=self.args.limit,
                device=self.args.device,
                cache_requests=None
        )
            
            if not results or 'results' not in results:
                raise BenchmarkError("Benchmark evaluation returned no results")
                
            results['results']['model_name'] = model_name
            return results
            
        except Exception as e:
            raise BenchmarkError(f"Benchmark failed for model {model_name}: {str(e)}") from e

    def run_benchmark(self, results_manager) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run benchmarks for both base and unlearned models.

        Args:
            results_manager: Manager instance for handling result storage and retrieval

        Returns:
            Tuple containing:
            - Dict: Base model evaluation results
            - Dict: Unlearned model evaluation results

        Note:
            If base model results exist and rerun_base is False, existing results will be loaded
            instead of running a new evaluation.
        """
        # base_model_path = results_manager.get_base_model_path()
        base_model_path = os.path.join(results_manager.results_path, "base_model_eval.json")
        if isinstance(base_model_path, str):
            base_model_path = Path(base_model_path)
        
        if base_model_path.exists() and not self.args.rerun_base:
            print("Base model evaluation already exists. Skipping base model evaluation.")
            base_results = results_manager.load_json(base_model_path)
        else:
            print("Running base model evaluation...")
            base_results = self.benchmark(self.args.model_name)
            results_manager.save_results(base_results, "base_model")

        print("Running unlearned model evaluation...")
        unlearned_results = self.benchmark(self.args.unlearned_model)
        results_manager.save_results(unlearned_results, "unlearned_model")

        return base_results, unlearned_results