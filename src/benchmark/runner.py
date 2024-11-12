from typing import Dict, Tuple
import lm_eval
import os

class BenchmarkRunner:
    def __init__(self, args):
        self.args = args

    def benchmark(self, model_name: str) -> Dict:
        """Run benchmark evaluation for a single model."""
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name},trust_remote_code={self.args.trust_remote_code}",
            tasks=self.args.benchmarks,
            log_samples=False,
            batch_size=self.args.batch_size,
            limit=self.args.limit,
            device=self.args.device,
            cache_requests=True,
            use_cache=os.path.join(os.path.dirname(os.path.dirname(self.args.unlearned_model)), "eval_cache")
        )
        results['results']['model_name'] = model_name
        return results

    def run_benchmark(self, results_manager) -> Tuple[Dict, Dict]:
        """Run benchmarks for both base and unlearned models."""
        base_model_path = results_manager.get_base_model_path()
        
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