import optuna
import json
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
import logging

# Add src directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from logger_config import setup_logger

logger = setup_logger()

def load_base_model_results(model_dir):
    """Load base model evaluation results"""
    base_results_path = model_dir / "base_model_eval.json"
    if not base_results_path.exists():
        logger.warning(f"No base model results found at {base_results_path}")
        return None
    
    try:
        with open(base_results_path) as f:
            data = json.load(f)
            
        # Debug information about the loaded data
        logger.debug("Base model results structure:")
        logger.debug(f"Keys in data: {list(data.keys())}")
        logger.debug(f"Keys in results: {list(data.get('results', {}).keys())}")
        
        # Extract results
        results = data.get('results', {})
        base_results = {
            'wmdp': results.get('wmdp', {}).get('acc,none', 0.0),
            'wmdp_bio': results.get('wmdp_bio', {}).get('acc,none', 0.0),
            'mmlu': results.get('mmlu', {}).get('acc,none', 0.0)
        }
        
        logger.info("Loaded base model results:")
        for metric, value in base_results.items():
            logger.info(f"  {metric}: {value}")
            
        return base_results
        
    except Exception as e:
        logger.error(f"Error loading base model results: {e}")
        return None

def load_existing_trials(model_dir):
    """Load completed trials from existing results files"""
    trials = []
    model_dir = Path(model_dir)
    logger.info(f"Looking for trials in {model_dir}")
    
    # Get base model results for comparison
    base_results = load_base_model_results(model_dir)
    
    # Find all trial directories (numbered folders)
    trial_dirs = list(model_dir.glob("[0-9]*"))
    logger.info(f"Found {len(trial_dirs)} potential trial directories")
    
    for trial_dir in trial_dirs:
        try:
            trial_num = int(trial_dir.name)
            results_file = trial_dir / "unlearned_model_eval.json"
            
            if not results_file.exists():
                logger.warning(f"No results file found for trial {trial_num}")
                continue
                
            logger.debug(f"Processing trial {trial_num}")
            
            # Load results
            with open(results_file) as f:
                results = json.load(f)
            
            # Debug information about the loaded data
            logger.debug(f"Trial {trial_num} results structure:")
            logger.debug(f"Keys in results: {list(results.keys())}")
            
            # Calculate metrics
            wmdp_bio_score = results.get('results', {}).get('wmdp_bio', {}).get('acc,none', 0.0)
            mmlu_score = results.get('results', {}).get('mmlu', {}).get('acc,none', 0.0)
            
            logger.debug(f"Trial {trial_num} raw scores:")
            logger.debug(f"  WMDP Bio: {wmdp_bio_score}")
            logger.debug(f"  MMLU: {mmlu_score}")
            
            # Calculate objective value
            baseline_mmlu = 0.45
            mmlu_penalty = max(0, baseline_mmlu - mmlu_score) * 2.0
            objective_value = wmdp_bio_score + mmlu_penalty
            
            logger.debug(f"Trial {trial_num} calculated metrics:")
            logger.debug(f"  MMLU Penalty: {mmlu_penalty}")
            logger.debug(f"  Objective Value: {objective_value}")
            
            # Create trial-like object
            trial_info = {
                'number': trial_num,
                'value': objective_value,
                'metrics': {
                    'wmdp_bio_acc': wmdp_bio_score,
                    'mmlu_acc': mmlu_score,
                    'mmlu_penalty': mmlu_penalty
                }
            }
            trials.append(trial_info)
            
        except Exception as e:
            logger.error(f"Error processing trial {trial_dir}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(trials)} trials")
    if trials:
        best_trial = max(trials, key=lambda x: x['value'])
        logger.info("\nBest trial metrics:")
        logger.info(f"  Trial number: {best_trial['number']}")
        logger.info(f"  Objective value: {best_trial['value']:.4f}")
        logger.info(f"  WMDP Bio accuracy: {best_trial['metrics']['wmdp_bio_acc']:.4f}")
        logger.info(f"  MMLU accuracy: {best_trial['metrics']['mmlu_acc']:.4f}")
        logger.info(f"  MMLU penalty: {best_trial['metrics']['mmlu_penalty']:.4f}")
    
    return trials, base_results

def create_visualizations(trials, base_results, output_dir):
    """Create visualization plots from trial data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial data
    with open(output_dir / 'current_results.json', 'w') as f:
        json.dump({
            'base_model': base_results,
            'trials': trials
        }, f, indent=2)
    
    # Sort trials by trial number
    trials = sorted(trials, key=lambda x: x['number'])
    
    # Create plots
    trial_numbers = [t['number'] for t in trials]
    values = [t['value'] for t in trials]
    wmdp_bio_scores = [t['metrics']['wmdp_bio_acc'] for t in trials]
    mmlu_scores = [t['metrics']['mmlu_acc'] for t in trials]
    mmlu_penalties = [t['metrics']['mmlu_penalty'] for t in trials]
    
    # Log the sorted data for debugging
    logger.debug("Sorted trial data:")
    for i, trial_num in enumerate(trial_numbers):
        logger.debug(f"Trial {trial_num}: "
                    f"Objective={values[i]:.4f}, "
                    f"WMDP Bio={wmdp_bio_scores[i]:.4f}, "
                    f"MMLU={mmlu_scores[i]:.4f}, "
                    f"Penalty={mmlu_penalties[i]:.4f}")
    
    # Combined metrics plot
    fig = go.Figure()
    
    # Add base model results as horizontal lines if available
    if base_results:
        for metric, value in base_results.items():
            if value != 0:  # Only add non-zero baseline values
                fig.add_hline(y=value, line_dash="dash", 
                             annotation_text=f"Base {metric}", 
                             annotation_position="right")
    
    # Add traces with sorted data
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=values,
        mode='lines+markers',
        name='Objective Value',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=wmdp_bio_scores,
        mode='lines+markers',
        name='WMDP Bio Accuracy',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=mmlu_scores,
        mode='lines+markers',
        name='MMLU Accuracy',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=mmlu_penalties,
        mode='lines+markers',
        name='MMLU Penalty',
        line=dict(width=2)
    ))
    
    # Update layout with better formatting
    fig.update_layout(
        title="Optimization History",
        xaxis_title="Trial Number",
        yaxis_title="Score",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        # Add grid for better readability
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            range=[0, max(max(values), max(wmdp_bio_scores), max(mmlu_scores)) * 1.1]
        ),
        plot_bgcolor='white'
    )
    
    # Save plots
    fig.write_html(output_dir / "optimization_history.html")
    fig.write_image(output_dir / "optimization_history.png")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to model directory containing trial results')
    parser.add_argument('--output_dir', type=str, default='partial_optimization_results',
                       help='Directory to save visualization results')
    args = parser.parse_args()
    
    logger.info("Loading existing trials...")
    trials, base_results = load_existing_trials(args.model_dir)
    logger.info(f"Found {len(trials)} completed trials")
    
    if base_results:
        logger.info("\nBase model results:")
        for metric, value in base_results.items():
            logger.info(f"  {metric}: {value}")
    
    logger.info("\nCreating visualizations...")
    create_visualizations(trials, base_results, args.output_dir)
    logger.info(f"\nVisualizations have been saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()