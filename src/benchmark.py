import argparse
import json
import os
import shutil
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import lm_eval
import torch

from plot import BenchmarkPlotter
from utils import load_yaml_config, CustomJSONEncoder


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
            device=self.args.device,
            trust_remote_code=True
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
        if self.current_subfolder is None:
            self.current_subfolder = max([f for f in os.listdir(self.results_path) if os.path.isdir(os.path.join(self.results_path, f)) and f.isdigit()])
        
        base_path = os.path.join(self.results_path, "base_model_eval.json")
        unlearned_path = os.path.join(self.results_path, self.current_subfolder, "unlearned_model_eval.json")

        with open(base_path, 'r') as f:
            base_results = json.load(f)
        with open(unlearned_path, 'r') as f:
            unlearned_results = json.load(f)

        return base_results, unlearned_results


    def plot_results(self,
                     datasets = ['wmdp_bio', 'wmdp_cyber', 'wmdp_chem', 'mmlu'],
                     display_names = ['WMDP-Bio', 'WMDP-Cyber', 'WMDP-Chem', 'MMLU']
    ):
        base_results, unlearned_results = self.load_results()
        model_name = base_results['results']['model_name']

        scores = {
            'Base': [self.get_accuracy(base_results, dataset) for dataset in datasets],
            'RMU (unlearned model)': [self.get_accuracy(unlearned_results, dataset) for dataset in datasets]
        }

        self.create_plot(scores, display_names, model_name)
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
        return self.plot_results(datasets=datasets, display_names=display_names)


    def create_plot(self, scores: Dict[str, List[float]], display_names: List[str], model_name: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(display_names))
        width = 0.35

        for i, (label, data) in enumerate(scores.items()):
            offset = width * (i - 0.5)
            ax.bar(x + offset, data, width, label=label, color=['#1f77b4', '#aec7e8'][i])

        ax.set_ylabel('Accuracy')
        ax.set_title(f'WMDP and MMLU Accuracy After Unlearning ({model_name.split("/")[-1]})')
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
        
        plot_path = os.path.join(self.results_path, self.current_subfolder, "accuracy_comparison.png")
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
    
    if args.plot_only:
        benchmarker.plot_results()
        benchmarker.plot_mmlu_subcategories()
    else:
        benchmarker.run_benchmark()
        benchmarker.plot_results()
        benchmarker.plot_mmlu_subcategories()

if __name__ == "__main__":
    main()