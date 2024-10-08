import argparse
import lm_eval
from lm_eval.utils import handle_non_serializable
import json

# TODO: handle local and non-local models

class BenchmarkModels:

    def __init__(self, args):
        self.args = args

  
    def benchmark(self):
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={self.model_name}",
            tasks=self.benchmarks,
            log_samples=False,
            batch_size=self.args.batch_size
        )
        results['results']['model_name'] = self.model_name
        return results


    def save_results(self,
                     results: dict
    ):
        """Save the evaluation results to a JSON file."""
        unlearned_model = "unlearned_model" if self.args.unlearned_model else "base_model"
        output_path = f"{self.args.results_path}/{unlearned_model}.json"
        with open(output_path, "w") as fp:
            json.dump(results, fp, indent=2, default=handle_non_serializable, ensure_ascii=False)
        print(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    benchmarker = BenchmarkModels(args.model)
    
    results = benchmarker.benchmark()
    benchmarker.save_results(results)