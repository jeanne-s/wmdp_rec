import argparse
import json
import os

import lm_eval
import torch

from utils import load_yaml_config, CustomJSONEncoder

# TODO: handle local and non-local models

# from lm_eval.utils import handle_non_serializable

class BenchmarkModels:

    def __init__(self, args):
        self.args = args

  
    def benchmark(self):
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={self.args.model_name}",
            tasks=self.args.benchmarks,
            log_samples=False,
            batch_size=self.args.batch_size,
            limit=self.args.limit,
            device=self.args.device
        )
        results['results']['model_name'] = self.args.model_name
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
            json.dump(results, fp, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

        print(f"Benchmark results saved to {output_path}")
    
    
    def plot_results(self):
        """
        Load and plot the results from the JSON file saved by save_results.
        """
        # Determine the model name and the base directory for loading results
        unlearned_model = "unlearned_model" if self.args.unlearned_model else "base_model"
        model_dir = os.path.join(self.args.results_path, unlearned_model)

        # Find the latest subfolder
        subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
        numbers = [int(f) for f in subfolders if f.isdigit()]
        latest_subfolder = f"{max(numbers):02}" if numbers else "00"

        # Construct the path to the latest results file
        results_path = os.path.join(model_dir, latest_subfolder, f"{unlearned_model}_results.json")

        # Load the results
        with open(results_path, 'r') as file:
            results = json.load(file)

        # Extract and print the results
        model_name = results['results']['model_name']
        wmdp = results['results']['wmdp']['acc,none']

        # TODO: Add plotting logic here


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (default: cuda)")
    parser.add_argument("--plot_only", action="store_true", help="Only plot results without running evaluation")
    args = parser.parse_args()

    # Check if CUDA is available, if not, use CPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = "cpu"

    config = load_yaml_config(file_path=args.config_file)
    
    for attr, value in vars(args).items():
        setattr(config, attr, value)

    benchmarker = BenchmarkModels(config)
    
    if args.plot_only:
        benchmarker.plot_results()
    else:
        results = benchmarker.benchmark()
        benchmarker.save_results(results)
        benchmarker.plot_results()
