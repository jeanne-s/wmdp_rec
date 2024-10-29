import os
import json
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List

class BenchmarkPlotter:
    def __init__(self, results_path: str, current_subfolder: str):
        self.results_path = results_path
        self.current_subfolder = current_subfolder

    @staticmethod
    def get_accuracy(results: Dict, task: str) -> float:
        return results['results'].get(task, {}).get('acc,none', 0) * 100

    def plot_results(self, base_results: Dict, unlearned_results: Dict,
                    datasets: List[str] = None,
                    display_names: List[str] = None):
        if datasets is None:
            datasets = ['wmdp_bio', 'wmdp_cyber', 'wmdp_chem', 'mmlu']
        if display_names is None:
            display_names = ['WMDP-Bio', 'WMDP-Cyber', 'WMDP-Chem', 'MMLU']

        model_name = base_results['results']['model_name']
        scores = {
            'Base': [self.get_accuracy(base_results, dataset) for dataset in datasets],
            'RMU': [self.get_accuracy(unlearned_results, dataset) for dataset in datasets]
        }

        self._create_plot(scores, display_names, model_name)
        self._print_results(base_results, unlearned_results)

    def plot_mmlu_subcategories(self, base_results: Dict, unlearned_results: Dict):
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
        self.plot_results(base_results, unlearned_results, datasets, display_names)

    def _create_plot(self, scores: Dict[str, List[float]], display_names: List[str], 
                    model_name: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(display_names))
        width = 0.35

        for i, (label, data) in enumerate(scores.items()):
            offset = width * (i - 0.5)
            ax.bar(x + offset, data, width, label=label, 
                  color=['#1f77b4', '#aec7e8'][i])

        self._format_plot(ax, x, display_names, model_name, scores)
        self._save_plot(fig)

    def _format_plot(self, ax, x, display_names: List[str], model_name: str, 
                    scores: Dict[str, List[float]]):
        ax.set_ylabel('Accuracy')
        ax.set_title(f'WMDP and MMLU Accuracy After Unlearning ({model_name.split("/")[-1]})')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.axhline(y=25, color='black', linestyle='--', linewidth=1, 
                  label='Random Chance')
        ax.legend()
        ax.set_ylim(0, 100)

        for i in range(len(display_names)):
            if scores['RMU'][i] > scores['Base'][i]:
                self._add_arrow(ax, i, max(scores['Base'][i], scores['RMU'][i]), '↑')
            elif scores['RMU'][i] < scores['Base'][i]:
                self._add_arrow(ax, i, min(scores['Base'][i], scores['RMU'][i]), '↓')

    def _save_plot(self, fig):
        plt.tight_layout()
        plot_path = os.path.join(self.results_path, self.current_subfolder, 
                                "accuracy_comparison.png")
        fig.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close(fig)

    @staticmethod
    def _add_arrow(ax, x, y, direction):
        ax.annotate(direction, xy=(x, y), 
                   xytext=(x, y + 2 if direction == '↑' else y - 2),
                   ha='center', va='bottom' if direction == '↑' else 'top', 
                   fontsize=12)

    @staticmethod
    def _print_results(base_results: Dict, unlearned_results: Dict):
        print("\nBase Model Results:")
        print(json.dumps(base_results['results'], indent=2))
        print("\nUnlearned Model Results:")
        print(json.dumps(unlearned_results['results'], indent=2))