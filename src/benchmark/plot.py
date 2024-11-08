import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class BenchmarkPlotter:
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)

    @staticmethod
    def get_accuracy(results: Dict, task: str) -> float:
        """Extract accuracy score for a specific task."""
        return results['results'].get(task, {}).get('acc,none', 0) * 100

    def plot_results(self,
                    base_results: Dict,
                    unlearned_results: Dict,
                    plot_title="WMDP and MMLU Accuracy After Unlearning",
                    datasets=['wmdp_bio', 'wmdp_cyber', 'wmdp_chem', 'mmlu'],
                    display_names=['WMDP-Bio', 'WMDP-Cyber', 'WMDP-Chem', 'MMLU'],
                    plot_filename="accuracy_comparison.png"
    ):
        """Create and save comparison plot."""
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

    def plot_mmlu_subcategories(self, base_results: Dict, unlearned_results: Dict):
        """Create and save MMLU subcategories plot."""
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
        self.plot_results(
            base_results=base_results,
            unlearned_results=unlearned_results,
            datasets=datasets, 
            display_names=display_names, 
            plot_filename="mmlu_subcategories.png",
            plot_title="MMLU Accuracy after Unlearning"
        )

    def create_plot(self, 
                   scores: Dict[str, List[float]], 
                   display_names: List[str], 
                   model_name: str,
                   plot_filename: str = "accuracy_comparison.png",
                   plot_title: str = "WMDP and MMLU Accuracy After Unlearning"
    ):
        """Create and save the plot."""
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
        
        plot_path = os.path.join(self.results_path, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.show()

    @staticmethod
    def add_arrow(ax, x, y, direction):
        """Add direction arrow to plot."""
        ax.annotate(direction, xy=(x, y), xytext=(x, y + 2 if direction == '↑' else y - 2),
                   ha='center', va='bottom' if direction == '↑' else 'top', fontsize=12)

    @staticmethod
    def print_results(base_results: Dict, unlearned_results: Dict):
        """Print results to console."""
        print("Base Model Results:")
        print(json.dumps(base_results['results'], indent=2))
        print("\nUnlearned Model Results:")
        print(json.dumps(unlearned_results['results'], indent=2))