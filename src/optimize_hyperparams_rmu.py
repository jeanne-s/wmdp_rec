import optuna
import subprocess
import json
import os
import sys
import numpy as np
import logging
import time
import argparse
from pathlib import Path
from transformers import AutoConfig
import yaml
import threading
from typing import Dict, Any, Tuple, Optional

# Add src directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from logger_config import setup_logger

# Replace existing logging setup with shared configuration
logger = setup_logger()

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for hyperparameter optimization.

    Returns:
        Namespace containing parsed arguments including:
        - model: Model name or path to config files
        - n_trials: Number of optimization trials to run
        - finetune_config: Path to finetuning configuration
        - benchmark_config: Path to benchmarking configuration
    """
    parser = argparse.ArgumentParser(description='Optimize hyperparameters for model unlearning')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str,
                      help='Model name - will use yaml_files/finetuning/{model}.yaml and yaml_files/benchmarking/{model}.yaml')
    group.add_argument('--finetune_config', type=str,
                      help='Path to base finetuning YAML configuration file')
    group.add_argument('--benchmark_config', type=str,
                      help='Path to base benchmarking YAML configuration file')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of optimization trials to run')
    args = parser.parse_args()

    # Set config paths based on model name if provided
    if args.model:
        args.finetune_config = f"yaml_files/finetuning/{args.model}.yaml"
        args.benchmark_config = f"yaml_files/benchmarking/{args.model}.yaml"
    
    # Load model name and seed from finetune config
    with open(args.finetune_config) as f:
        config = yaml.safe_load(f)
        args.model_name = config['args']['model_name']
        args.seed = config['args'].get('seed', 42)  # Default to 42 if not specified
    
    # Get model architecture info
    model_config = AutoConfig.from_pretrained(args.model_name)
    args.num_layers = model_config.num_hidden_layers
    args.hidden_size = model_config.hidden_size
    
    logger.info(f"Model {args.model_name} has {args.num_layers} layers")
    logger.info(f"Hidden size: {args.hidden_size}")
    logger.info(f"Using seed: {args.seed}")
    
    return args

def find_last_complete_trial(model_dir: Path) -> int:
    """Find the last trial number that has complete benchmark results"""
    if not model_dir.exists():
        return -1
        
    last_complete = -1
    for trial_dir in sorted(model_dir.glob("[0-9]*")):  # Get all numeric directories
        try:
            trial_num = int(trial_dir.name)
            eval_file = trial_dir / "unlearned_model_eval.json"
            if eval_file.exists() and eval_file.stat().st_size > 0:
                last_complete = trial_num
            else:
                # Found first incomplete trial
                break
        except ValueError:
            continue
    
    return last_complete

def run_command(command, timeout=7200):
    """Run command with enhanced error handling, logging, and real-time output"""
    logger.info(f"Executing command: {command}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_lines = []
        last_line_length = 0
        
        def handle_output(pipe, lines):
            nonlocal last_line_length
            for line in pipe:
                line = line.rstrip()
                if any(line.startswith(f"{i}%") for i in range(101)):
                    # Clear the current line and print the progress
                    print(f'\033[2K\r{line}', end='', flush=True)
                    last_line_length = len(line)
                else:
                    # For non-progress lines, clear current line and print with newline
                    if last_line_length > 0:
                        print('\033[2K\r', end='')  # Clear the current line
                        last_line_length = 0
                    print(line, flush=True)
                lines.append(line)
        
        stdout_thread = threading.Thread(target=handle_output, args=(process.stdout, stdout_lines))
        stdout_thread.daemon = True
        stdout_thread.start()
        
        process.wait(timeout=timeout)
        stdout_thread.join()
        
        # Ensure we're on a new line after any progress bars
        if last_line_length > 0:
            print()
            
        execution_time = time.time() - start_time
        logger.info(f"Command completed in {execution_time:.2f} seconds")
        
        return stdout_lines, [], process.returncode
        
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"Command timed out after {timeout} seconds")
        raise
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        raise

def verify_file_exists(filepath, timeout=300, check_interval=10):
    """Wait for file to exist with timeout"""
    filepath = Path(filepath)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if filepath.exists():
            size = filepath.stat().st_size
            if size > 0:
                logger.info(f"File {filepath} exists and is non-empty (size: {size} bytes)")
                return True
            else:
                logger.warning(f"File {filepath} exists but is empty (size: {size} bytes)")
        else:
            logger.debug(f"File {filepath} does not exist yet, waiting...")
        time.sleep(check_interval)
    
    logger.error(f"Timeout waiting for file {filepath} after {timeout} seconds")
    return False

def get_results(results_file):
    """Get results from evaluation output"""
    with open(results_file) as f:
        data = json.load(f)
    
    # Add logging to debug the structure
    logger.debug(f"Results file contents: {json.dumps(data, indent=2)}")
    
    # Check if we have the expected structure
    required_keys = ['results']
    for key in required_keys:
        if key not in data:
            # Create the expected structure if it doesn't exist
            if key == 'results':
                # The file already has the results directly in the root
                return {
                    'wmdp': data.get('wmdp', {}).get('acc,none', 0.0),
                    'wmdp_bio': data.get('wmdp_bio', {}).get('acc,none', 0.0),
                    'mmlu': data.get('mmlu', {}).get('acc,none', 0.0)
                }
            raise KeyError(f"Missing required key {key} in results")

    # If we have the expected structure, proceed as before
    results = data['results']
    return {
        'wmdp': results.get('wmdp', {}).get('acc,none', 0.0),
        'wmdp_bio': results.get('wmdp_bio', {}).get('acc,none', 0.0),
        'mmlu': results.get('mmlu', {}).get('acc,none', 0.0)
    }

def get_module_params(param_type, layer_id):
    """Return parameter indices based on type and layer"""
    if param_type not in ["mlp", "attention", "both"]:
        logger.error(f"Invalid param_type: {param_type}")
        raise ValueError(f"Invalid param_type: {param_type}")
    
    return str(layer_id)

def create_trial_configs(
    trial: optuna.Trial,
    base_finetune_config: Dict[str, Any],
    base_benchmark_config: Dict[str, Any],
    trial_dir: Path,
    params: Dict[str, Any]
) -> Tuple[Path, Path]:
    """
    Create configuration files for a specific optimization trial.

    Args:
        trial: Current optimization trial
        base_finetune_config: Base configuration for finetuning
        base_benchmark_config: Base configuration for benchmarking
        trial_dir: Directory to store trial configurations
        params: Trial-specific parameters

    Returns:
        Tuple containing paths to the created config files (finetune_config_path, benchmark_config_path)
    """
    # Create finetuning config
    with open(base_finetune_config) as f:
        finetune_config = yaml.safe_load(f)
    
    model_name = finetune_config['args']['model_name']
    model_name_safe = model_name.replace('/', '_')

    model_dir = trial_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Update only the parameters we're sweeping
    finetune_config['args'].update({
        'steering_coefficient': params['steering_coeff'],
        'alpha': params['alpha'],
        'forget_layer_id': params['layer_id'],
        'optimizer_param_layer_id': [params['param_ids']],
        'update_layer_ids': list(range(max(0, params['layer_id'] - 2), params['layer_id'] + 1))
    })
    
    # Keep the output path update
    finetune_config['args']['updated_model_path'] = str(trial_dir)
    
    finetune_config_path = model_dir / "finetune_config.yaml"
    with open(finetune_config_path, 'w') as f:
        yaml.dump(finetune_config, f)
    
    # Create benchmarking config
    with open(base_benchmark_config) as f:
        benchmark_config = yaml.safe_load(f)
    
    # Update paths to put results inside trial number folder
    model_path = model_dir / f"{trial.number:02d}" / "model.pt"
    benchmark_config['args'].update({
        'model_name': model_name,
        'unlearned_model': str(model_path),
        'results_path': str(model_dir),
        'trial': f"{trial.number:02d}"
    })
    
    benchmark_config_path = model_dir / "benchmark_config.yaml"
    with open(benchmark_config_path, 'w') as f:
        yaml.dump(benchmark_config, f)
    
    return finetune_config_path, benchmark_config_path

def suggest_hyperparameters(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a trial in the optimization study.
    Only sweeps specific parameters while keeping others constant from config.

    Args:
        trial: Current optimization trial
        args: Configuration arguments containing model info

    Returns:
        Dictionary containing suggested hyperparameter values for:
        - layer_id: Layer ID for operations
        - alpha: Weight for retain loss
        - steering_coeff: Coefficient for steering vector
        - param_ids: Parameter IDs to optimize
    """
    params = {
        'layer_id': trial.suggest_int('layer_id', 2, args.num_layers),  # Use actual number of layers
        'alpha': trial.suggest_float('alpha', 300.0, 2000.0),
        'steering_coeff': trial.suggest_float('steering_coeff', 1.0, 300.0, log=True),
        'param_ids': trial.suggest_int('param_ids', 0, 8),  # Also adjust this range
    }
    
    return params

def evaluate_trial(
    trial: optuna.Trial,
    args: argparse.Namespace,
    params: Dict[str, Any]
) -> float:
    """
    Evaluate a single optimization trial.

    Args:
        trial: Current optimization trial
        args: Configuration arguments
        params: Hyperparameters to evaluate

    Returns:
        float: Evaluation metric (lower is better)

    Raises:
        optuna.TrialPruned: If trial is pruned early
    """
    trial_start_time = time.time()
    logger.info(f"\nStarting trial {trial.number}")
    
    try:
        logger.info(f"Trial {trial.number} parameters: {json.dumps(params, indent=2)}")
        
        # Create trial directory without redundant subfolder
        trial_dir = Path("models")
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load baseline MMLU score
        base_eval_file = trial_dir / args.model_name / "base_model_eval.json"
        baseline_mmlu = None
        if base_eval_file.exists():
            try:
                with open(base_eval_file) as f:
                    base_data = json.load(f)
                baseline_mmlu = base_data.get('results', {}).get('mmlu', {}).get('acc,none')
                logger.info(f"Loaded baseline MMLU score: {baseline_mmlu}")
            except Exception as e:
                logger.warning(f"Failed to load baseline MMLU score: {e}")
        
        # Create configs and run commands
        finetune_config, benchmark_config = create_trial_configs(
            trial, 
            args.finetune_config, 
            args.benchmark_config, 
            trial_dir, 
            params
        )
        
        # Run unlearning
        unlearn_command = f"python src/rmu.py --config_file {finetune_config}"
        stdout, stderr, return_code = run_command(unlearn_command)

        case_dir = trial_dir / args.model_name / f"{trial.number:02d}"
        
        # Verify model files exist - update path to use model_name_safe
        model_file = case_dir / "model.pt" / "model.safetensors"
        if not verify_file_exists(model_file, timeout=10):
            raise FileNotFoundError(f"Model file {model_file} not created")
        
        # Update benchmark config to use correct model path
        with open(benchmark_config) as f:
            benchmark_cfg = yaml.safe_load(f)
        benchmark_cfg['args']['unlearned_model'] = str(model_file.parent)  # Point to the model.pt directory
        with open(benchmark_config, 'w') as f:
            yaml.dump(benchmark_cfg, f)
        
        # Run evaluation
        eval_command = f"python src/benchmark.py --config_file {benchmark_config} --device cuda"
        stdout, stderr, return_code = run_command(eval_command)
        
        # Update results file path to be inside trial folder
        results_file = case_dir / "unlearned_model_eval.json"
        
        # Add additional verification for results file
        if not verify_file_exists(results_file, timeout=10):
            raise FileNotFoundError(f"Results file {results_file} not created")
        
        results = get_results(results_file)
        
        # Calculate objective value
        objective_value = results['wmdp_bio']
        if baseline_mmlu is not None:
            mmlu_penalty = max(0, baseline_mmlu - results['mmlu']) * 2.0
            objective_value += mmlu_penalty
            trial.set_user_attr('mmlu_penalty', mmlu_penalty)
        else:
            logger.info("No baseline MMLU score available - skipping penalty calculation")
            trial.set_user_attr('mmlu_penalty', 0.0)
        
        # Store metrics
        trial.set_user_attr('wmdp_bio_acc', results['wmdp_bio'])
        trial.set_user_attr('mmlu_acc', results['mmlu'])
        
        logger.info(f"Trial {trial.number} completed in {time.time() - trial_start_time:.2f} seconds")
        logger.info(f"Trial {trial.number} objective value: {objective_value}")
        
        return objective_value
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
        # if trial_dir.exists():
        #     import shutil
        #     shutil.rmtree(trial_dir)
        raise

def create_study(study_name: str, storage_name: str) -> optuna.Study:
    """Create or load an Optuna study."""
    try:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True
        )
        
        # Add custom sampling strategy
        study.sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=42
        )
        
        return study
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create optimization study: {str(e)}") from e

def save_optimization_results(study: optuna.Study) -> None:
    """Save optimization results to file."""
    try:
        os.makedirs('optimization_results', exist_ok=True)
        
        best_trial = study.best_trial
        if not best_trial:
            raise ValueError("No completed trials found in study")
            
        study_stats = {
            'best_trial': {
                'value': best_trial.value,
                'params': best_trial.params,
                'user_attrs': best_trial.user_attrs
            },
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'user_attrs': t.user_attrs
                }
                for t in study.trials if t.value is not None
            ]
        }
        
        output_path = Path('optimization_results') / 'study_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(study_stats, f, indent=2)
            
        logger.info(f"Optimization results saved to {output_path}")
        
    except Exception as e:
        raise ConfigurationError(f"Failed to save optimization results: {str(e)}") from e

def main():
    args = parse_args()
    
    model_name_safe = args.model_name.replace('/', '_')
    study_name = f"{model_name_safe}_unlearning_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    # Find the last complete trial
    model_dir = Path("models") / args.model_name
    last_complete_trial = find_last_complete_trial(model_dir)
    logger.info(f"Last complete trial: {last_complete_trial}")
    
    # Create a new study with a different name to avoid the incomplete trials
    new_study_name = f"{study_name}_{int(time.time())}"
    new_storage_name = f"sqlite:///{new_study_name}.db"
    
    # Create the new study
    new_study = optuna.create_study(
        study_name=new_study_name,
        storage=new_storage_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Load the old study
    try:
        old_study = optuna.load_study(
            study_name=study_name,
            storage=storage_name
        )
        
        # Copy completed trials to the new study
        for trial in old_study.trials:
            if trial.number <= last_complete_trial:
                new_study.add_trial(trial)
    except Exception as e:
        logger.warning(f"Could not load old study: {e}")
    
    # Calculate remaining trials
    completed_trials = last_complete_trial + 1  # Convert to count
    remaining_trials = args.n_trials - completed_trials
    
    if remaining_trials <= 0:
        logger.info(f"All {args.n_trials} trials have been completed successfully. Exiting.")
        save_optimization_results(new_study)
        return
    
    logger.info(f"Completed trials: {completed_trials}")
    logger.info(f"Remaining trials: {remaining_trials}")
    
    # Update sampling strategy
    new_study.sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
        seed=args.seed
    )
    
    # Run remaining trials with the new study
    new_study.optimize(
        lambda trial: evaluate_trial(
            trial,
            args,
            suggest_hyperparameters(trial, args)
        ), 
        n_trials=remaining_trials
    )
    
    save_optimization_results(new_study)

def create_visualizations(study: optuna.Study) -> None:
    """
    Create visualization plots for optimization results.

    Args:
        study: Completed optimization study

    Creates:
        - Parameter importance plot
        - Optimization history plot
        - Parameter correlation plot
        Saved in 'optimization_results' directory
    """
    import plotly.graph_objects as go
    
    # Parameter importance plot
    param_importance = optuna.importance.get_param_importances(study)
    fig = go.Figure([go.Bar(
        x=list(param_importance.keys()),
        y=list(param_importance.values())
    )])
    fig.update_layout(title="Parameter Importance")
    fig.write_html("optimization_results/parameter_importance.html")
    
    # Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html("optimization_results/optimization_history.html")
    
    # Parallel coordinate plot
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html("optimization_results/parallel_coordinate.html")
    
    # Slice plot
    fig = optuna.visualization.plot_slice(study)
    fig.write_html("optimization_results/slice_plot.html")

if __name__ == "__main__":
    main()