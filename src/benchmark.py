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
                     results: dict):
        """
        Save the evaluation results to a JSON file. If the folder exists, 
        it creates subfolders with incremental numbers (e.g., 00, 01, etc.)
        for versioning the results.
        """
        # Determine the model name and the base directory for saving results
        unlearned_model = "unlearned_model" if self.args.unlearned_model else "base_model"
        model_dir = os.path.join(self.args.results_path, unlearned_model)

        # If the folder exists, create subfolders with incremental numbers (e.g., 00, 01, etc.)
        if os.path.exists(model_dir):
            subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
            numbers = [int(f) for f in subfolders if f.isdigit()]
            new_subfolder_num = f"{max(numbers) + 1:02}" if numbers else "00"
        else:
            # If folder doesn't exist, create the main folder and start with subfolder 00
            os.makedirs(model_dir)
            new_subfolder_num = "00"

        # Create the subfolder path
        output_dir = os.path.join(model_dir, new_subfolder_num)
        os.makedirs(output_dir, exist_ok=True)

        # Create the output file path
        output_path = os.path.join(output_dir, f"{unlearned_model}_results.json")

        # Save the results to the output file
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