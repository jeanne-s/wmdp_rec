"""
Main benchmarking orchestrator module.

This module provides the high-level interface for running benchmarks on models.
It coordinates between the benchmark runner, results manager, and plotter components
to provide a complete benchmarking pipeline.
"""

import os
from typing import Dict
import torch
import json
from benchmark.runner import BenchmarkRunner
from benchmark.results import ResultsManager
from benchmark.plot import BenchmarkPlotter
from exceptions import BenchmarkError, ConfigurationError
from pathlib import Path

import argparse
import torch
from utils import load_yaml_config


class BenchmarkModels:
    """
    High-level benchmarking orchestrator class.

    This class coordinates between the benchmark runner, results manager, and plotter
    to provide a complete benchmarking pipeline.

    Attributes:
        args: Configuration arguments for benchmarking
        results_path (str): Path to store benchmark results
        runner (BenchmarkRunner): Runner for executing benchmarks
        results_manager (ResultsManager): Manager for handling result storage
        plotter (BenchmarkPlotter): Plotter for generating visualizations
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.results_path = os.path.join("benchmark_results", self.args.model_name.split("/")[-1])
        
        self.runner = BenchmarkRunner(args)
        self.results_manager = ResultsManager(self.results_path)
        self.plotter = BenchmarkPlotter(self.results_path)

    def run(self) -> None:
        """
        Execute the complete benchmarking pipeline.
        
        This includes:
        - Running benchmarks if needed
        - Loading existing results
        - Generating comparison plots
        
        Raises:
            ConfigurationError: If configuration is invalid
            BenchmarkError: If benchmark execution fails
            ModelLoadError: If model loading fails
        """
        # try:
        if self.args.plot_only:
            base_results, unlearned_results = self.results_manager.load_results()
            self.plotter.plot_results(base_results, unlearned_results)
            self.plotter.plot_mmlu_subcategories(base_results, unlearned_results)
        else:
            # Validate configuration before running
            if not hasattr(self.args, 'model_name') or not self.args.model_name:
                raise ConfigurationError("model_name is required in configuration")
            if not hasattr(self.args, 'unlearned_model') or not self.args.unlearned_model:
                raise ConfigurationError("unlearned_model path is required in configuration")
            
            # Run benchmarks
            base_results, unlearned_results = self.runner.run_benchmark(self.results_manager)
            
            # Generate plots
            self.plotter.plot_results(base_results, unlearned_results)
            self.plotter.plot_mmlu_subcategories(base_results, unlearned_results)
                
        # except FileNotFoundError as e:
        #     raise BenchmarkError(f"Required file not found: {e}")
        # except json.JSONDecodeError as e:
        #     raise BenchmarkError(f"Invalid JSON in results file: {e}")
        # except Exception as e:
        #     raise BenchmarkError(f"Benchmark pipeline failed: {e}") from e

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