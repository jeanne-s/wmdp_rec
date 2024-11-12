import json
import os
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exceptions import BenchmarkError, ConfigurationError

logger = logging.getLogger(__name__)

class BenchmarkPlotter:
    """
    Handles visualization of benchmark results.

    This class creates various plots to compare performance between base and unlearned models,
    including overall accuracy comparisons and detailed MMLU subcategory analysis.

    Attributes:
        results_path (str): Path where plots will be saved
        figure_size (Tuple[int, int]): Default size for generated plots
        dpi (int): Resolution for saved plots
    """

    def __init__(self, results_path: str, figure_size: Tuple[int, int] = (10, 6), dpi: int = 300) -> None:
        """
        Initialize the plotter with configuration settings.

        Args:
            results_path: Directory where plots will be saved
            figure_size: Width and height of the plots in inches
            dpi: Dots per inch for plot resolution
        """
        self.results_path = results_path
        self.figure_size = figure_size
        self.dpi = dpi

    @staticmethod
    def get_accuracy(results: Dict[str, Any], task: str) -> float:
        """
        Extract accuracy score for a specific task from results dictionary.

        Args:
            results: Dictionary containing benchmark results
            task: Name of the task to get accuracy for (e.g., 'wmdp', 'mmlu')

        Returns:
            float: Accuracy score as a percentage (0-100)

        Note:
            Returns 0 if the task or accuracy score is not found in results.
            Converts raw accuracy (0-1) to percentage (0-100).
        """
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

        self.create_plot(
            scores=scores, 
            labels=display_names, 
            plot_filename=plot_filename,
            plot_title=plot_title,
            model_name=model_name
        )
        self.print_results(base_results, unlearned_results)

    def plot_mmlu_subcategories(
        self, 
        base_results: Dict, 
        unlearned_results: Dict,
        min_diff: float = 0.0
    ) -> None:
        """
        Create detailed plots comparing MMLU subcategory performance.

        Args:
            base_results: Dictionary containing base model evaluation results
            unlearned_results: Dictionary containing unlearned model evaluation results
            min_diff: Minimum difference threshold for highlighting changes

        Creates:
            - Bar plot comparing subcategory accuracies
            - Saves plot as 'mmlu_subcategories.png'
        """
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

    def create_plot(
        self,
        scores: Dict[str, List[float]],
        labels: List[str],
        plot_filename: str,
        plot_title: str,
        model_name: str
    ) -> None:
        """Create and save benchmark comparison plot."""
        try:
            if not scores or not labels:
                raise ValueError("Empty scores or labels provided")
                
            if len(scores['Base']) != len(scores['RMU (unlearned model)']):
                raise ValueError("Mismatched number of scores between models")
                
            if len(labels) != len(scores['Base']):
                raise ValueError("Number of labels doesn't match number of scores")
                
            plt.figure(figsize=self.figure_size)
            ax = plt.gca()
            
            x = range(len(labels))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], scores['Base'], width, 
                   label='Base', color='blue', alpha=0.6)
            ax.bar([i + width/2 for i in x], scores['RMU (unlearned model)'], width,
                   label='RMU (unlearned model)', color='red', alpha=0.6)
            
            # Add arrows for score changes
            for i in x:
                if scores['RMU (unlearned model)'][i] != scores['Base'][i]:
                    self.add_arrow(ax, i, 
                                 max(scores['Base'][i], scores['RMU (unlearned model)'][i]),
                                 '↑' if scores['RMU (unlearned model)'][i] > scores['Base'][i] else '↓')
            
            # Format the plot
            format_plot(ax, f"{plot_title}\n{model_name}", "Tasks", "Accuracy (%)")
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            plot_path = Path(self.results_path) / plot_filename
            plt.tight_layout()
            plt.savefig(plot_path, dpi=self.dpi)
            logger.info(f"Plot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            raise BenchmarkError(f"Failed to create plot: {str(e)}") from e

    @staticmethod
    def add_arrow(
        ax: plt.Axes, 
        x: float, 
        y: float, 
        direction: str
    ) -> None:
        """
        Add a directional arrow annotation to the plot.

        Args:
            ax: Matplotlib axes object to add the arrow to
            x: X-coordinate for the arrow
            y: Y-coordinate for the arrow
            direction: Arrow direction ('↑' or '↓')
        """
        ax.annotate(direction, xy=(x, y), xytext=(x, y + 2 if direction == '↑' else y - 2),
                   ha='center', va='bottom' if direction == '↑' else 'top', fontsize=12)

    def print_results(self, base_results: Dict[str, Any], unlearned_results: Dict[str, Any]) -> None:
        """
        Print formatted benchmark results to console.

        Args:
            base_results: Dictionary containing base model evaluation results
            unlearned_results: Dictionary containing unlearned model evaluation results

        Prints:
            - Base model results for each task
            - Unlearned model results for each task
            - Performance changes between base and unlearned models

        Example output:
            Base Model Results:
            WMDP: 75.3%
            MMLU: 52.1%

            Unlearned Model Results:
            WMDP: 74.8%
            MMLU: 51.9%
        """

    def get_mmlu_scores(self, results: Dict) -> Dict[str, float]:
        """
        Extract MMLU subcategory scores from results.

        Args:
            results: Dictionary containing evaluation results

        Returns:
            Dictionary mapping subcategory names to accuracy scores
        """

def format_plot(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: bool = True,
    grid: bool = True
) -> None:
    """
    Apply consistent formatting to a plot.

    Args:
        ax: Matplotlib axes object to format
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        grid: Whether to show grid
    """
    ax.set_title(title, pad=20, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if legend:
        ax.legend(fontsize=10)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)

def calculate_plot_metrics(
    base_results: Dict[str, Any],
    unlearned_results: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate metrics for plotting benchmark comparisons.

    Args:
        base_results: Results from base model evaluation
        unlearned_results: Results from unlearned model evaluation

    Returns:
        Tuple containing:
        - Dictionary of base model metrics
        - Dictionary of unlearned model metrics
    """