"""
Benchmark module for evaluating model performance.

This module provides functionality to evaluate and compare the performance of base and 
unlearned models across various benchmarks including WMDP and MMLU. It handles:
- Model evaluation using lm-eval
- Results storage and management
- Performance visualization through plots
- Comparative analysis of model performance
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import lm_eval
import torch

from utils import load_yaml_config, CustomJSONEncoder
from exceptions import ConfigurationError, BenchmarkError, ModelLoadError


class BenchmarkModels:
    """
    Main class for running model benchmarks and managing results.

    This class handles the complete benchmarking pipeline including:
    - Running evaluations on base and unlearned models
    - Saving and loading benchmark results
    - Generating comparison plots
    - Managing result directories and versioning

    Attributes:
        args: Configuration arguments for benchmarking
        results_path (str): Path to store benchmark results
        current_subfolder (Optional[str]): Current results subfolder being used
    """

    def __init__(self, args) -> None:
        self.args = args
        self.results_path = args.results_path if hasattr(args, 'results_path') else os.path.join("benchmark_results", self.args.model_name.split("/")[-1])
        #self.current_subfolder: Optional[str] = None
        self.current_subfolder = getattr(args, 'trial', None)
        # self.optimization = hasattr(args, 'optimization') and args.optimization

    def benchmark(self, model_name: str) -> Dict[str, Any]:
        """
        Run benchmark evaluation for a specific model.

        Args:
            model_name: Name or path of the model to evaluate

        Returns:
            Dictionary containing evaluation results with metrics for each task
        """
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
        results['results']['model_name'] = model_name
        return results


    def get_new_subfolder(self) -> str:
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            return "00"
        subfolders = [f for f in os.listdir(self.results_path) if os.path.isdir(os.path.join(self.results_path, f))]
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
        print(f"Benchmark results saved to {output_path}")

        if model_type == "base_model":
            main_output_path = os.path.join(self.results_path, "base_model_eval.json")
            shutil.copy2(output_path, main_output_path)
            print(f"Base model results also saved/updated at {main_output_path}")


    def run_benchmark(self) -> tuple[Dict, Dict]:
        base_model_path = os.path.join(self.results_path, "base_model_eval.json")
        
        if os.path.exists(base_model_path) and not self.args.rerun_base:
            print("Base model evaluation already exists. Skipping base model evaluation.")
            with open(base_model_path, 'r') as f:
                base_results = json.load(f)
        else:
            print("Running base model evaluation...")
            base_results = self.benchmark(self.args.model_name)
            self.save_results(base_results, "base_model")

        print("Running unlearned model evaluation...")
        unlearned_results = self.benchmark(self.args.unlearned_model)
        self.save_results(unlearned_results, "unlearned_model")

        return base_results, unlearned_results


    @staticmethod
    def get_accuracy(results: Dict, task: str) -> float:
        return results['results'].get(task, {}).get('acc,none', 0) * 100


    def load_results(self) -> tuple[Dict, Dict]:
        try:
            if self.current_subfolder is None:
                subfolders = [f for f in os.listdir(self.results_path) 
                             if os.path.isdir(os.path.join(self.results_path, f)) and f.isdigit()]
                if not subfolders:
                    raise BenchmarkError("No result folders found")
                self.current_subfolder = max(subfolders)
            
            base_path = os.path.join(self.results_path, "base_model_eval.json")
            unlearned_path = os.path.join(self.results_path, self.current_subfolder, "unlearned_model_eval.json")

            if not os.path.exists(base_path):
                raise BenchmarkError(f"Base model results not found at {base_path}")
            if not os.path.exists(unlearned_path):
                raise BenchmarkError(f"Unlearned model results not found at {unlearned_path}")

            try:
                with open(base_path, 'r') as f:
                    base_results = json.load(f)
                with open(unlearned_path, 'r') as f:
                    unlearned_results = json.load(f)
            except json.JSONDecodeError as e:
                raise BenchmarkError(f"Invalid JSON in results file: {str(e)}")

            return base_results, unlearned_results
        except Exception as e:
            raise BenchmarkError(f"Failed to load results: {str(e)}") from e


    def plot_results(self,
                     plot_title = "WMDP and MMLU Accuracy After Unlearning",
                     datasets = ['wmdp_bio', 'wmdp_cyber', 'wmdp_chem', 'mmlu'],
                     display_names = ['WMDP-Bio', 'WMDP-Cyber', 'WMDP-Chem', 'MMLU'],
                     plot_filename="accuracy_comparison.png"
    ):
        base_results, unlearned_results = self.load_results()
        model_name = base_results['results']['model_name']

        scores = {
            'Base': [self.get_accuracy(base_results, dataset) for dataset in datasets],
            'RMU (unlearned model)': [self.get_accuracy(unlearned_results, dataset) for dataset in datasets]
        }

        self.create_plot(scores, 
                         display_names, 
                         model_name, 
                         plot_filename=plot_filename,
                         plot_title=plot_title)
        self.print_results(base_results, unlearned_results)
        return


    def plot_mmlu_subcategories(self):
        datasets = [
            'mmlu', 
            'mmlu_college_computer_science', 
            'mmlu_computer_security', 
            'mmlu_college_biology',
            'mmlu_virology'
        ]
        display_names = [
            'All', 
            'College CS', 
            'Computer Security', 
            'College Biology',
            'Virology'
        ]
        return self.plot_results(datasets=datasets, 
                                 display_names=display_names, 
                                 plot_filename="mmlu_subcategories.png",
                                 plot_title="MMLU Accuracy after Unlearning")


    def create_plot(
        self, 
        scores: Dict[str, List[float]], 
        display_names: List[str], 
        model_name: str,
        plot_filename: str = "accuracy_comparison.png",
        plot_title: str = "WMDP and MMLU Accuracy After Unlearning"
    ) -> None:
        """
        Create and save a bar plot comparing model performances.

        Args:
            scores: Dictionary mapping model types to lists of accuracy scores
            display_names: List of benchmark names for x-axis labels
            model_name: Name of the model being evaluated
            plot_filename: Name of the output plot file
            plot_title: Title to display on the plot

        Note:
            The plot is saved in the current results subfolder and also displayed
            using plt.show()
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(display_names))
        width = 0.35

        for i, (label, data) in enumerate(scores.items()):
            offset = width * (i - 0.5)
            ax.bar(x + offset, data, width, label=label, color=['#1f77b4', '#aec7e8'][i])

        ax.set_ylabel('Accuracy')
        ax.set_title(f'{plot_title} ({model_name.split("/")[-1]})')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.axhline(y=25, color='black', linestyle='--', linewidth=1, label='Random Chance')
        ax.legend()

        for i in range(len(display_names)):
            if scores['RMU (unlearned model)'][i] > scores['Base'][i]:
                self.add_arrow(ax, i, max(scores['Base'][i], scores['RMU (unlearned model)'][i]), '↑')
            elif scores['RMU (unlearned model)'][i] < scores['Base'][i]:
                self.add_arrow(ax, i, min(scores['Base'][i], scores['RMU (unlearned model)'][i]), '↓')

        ax.set_ylim(0, 100)
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_path, self.current_subfolder, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.show()


    @staticmethod
    def add_arrow(ax, x, y, direction):
        ax.annotate(direction, xy=(x, y), xytext=(x, y + 2 if direction == '↑' else y - 2),
                    ha='center', va='bottom' if direction == '↑' else 'top', fontsize=12)


    @staticmethod
    def print_results(base_results: Dict, unlearned_results: Dict):
        print("Base Model Results:")
        print(json.dumps(base_results['results'], indent=2))
        print("\nUnlearned Model Results:")
        print(json.dumps(unlearned_results['results'], indent=2))


def main():
    try:
        parser = argparse.ArgumentParser(description="Model evaluation.")
        parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file")
        parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (default: cuda)")
        parser.add_argument("--plot_only", action="store_true", help="Only plot results without running evaluation")
        parser.add_argument("--rerun_base", action="store_true", help="Re-run base model evaluation even if it exists")
        args = parser.parse_args()

        if not os.path.exists(args.config_file):
            raise ConfigurationError(f"Config file not found: {args.config_file}")

        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            args.device = "cpu"

        config = load_yaml_config(file_path=args.config_file)
        for attr, value in vars(args).items():
            setattr(config, attr, value)

        benchmarker = BenchmarkModels(config)
        
        if args.plot_only:
            benchmarker.plot_results()
            benchmarker.plot_mmlu_subcategories()
        else:
            benchmarker.run_benchmark()
            benchmarker.plot_results()
            benchmarker.plot_mmlu_subcategories()

    except (ConfigurationError, BenchmarkError, ModelLoadError) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()