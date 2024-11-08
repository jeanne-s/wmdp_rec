import os
from typing import Dict
import torch
from benchmark.runner import BenchmarkRunner
from benchmark.results import ResultsManager
from benchmark.plot import BenchmarkPlotter

import argparse
import torch
from utils import load_yaml_config


class BenchmarkModels:
    def __init__(self, args):
        self.args = args
        self.results_path = os.path.join("benchmark_results", self.args.model_name.split("/")[-1])
        
        self.runner = BenchmarkRunner(args)
        self.results_manager = ResultsManager(self.results_path)
        self.plotter = BenchmarkPlotter(self.results_path)

    def run(self):
        if self.args.plot_only:
            base_results, unlearned_results = self.results_manager.load_results()
        else:
            base_results, unlearned_results = self.runner.run_benchmark(self.results_manager)
            
        self.plotter.plot_results(base_results, unlearned_results)
        self.plotter.plot_mmlu_subcategories(base_results, unlearned_results)
    

def main():
    parser = argparse.ArgumentParser(description="Model evaluation.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (default: cuda)")
    parser.add_argument("--plot_only", action="store_true", help="Only plot results without running evaluation")
    parser.add_argument("--rerun_base", action="store_true", help="Re-run base model evaluation even if it exists")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = "cpu"

    config = load_yaml_config(file_path=args.config_file)
    for attr, value in vars(args).items():
        setattr(config, attr, value)

    benchmarker = BenchmarkModels(config)
    benchmarker.run()

if __name__ == "__main__":
    main()