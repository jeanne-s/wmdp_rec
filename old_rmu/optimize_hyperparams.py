import optuna
import subprocess
import json
import os
import numpy as np
import logging
import time
from pathlib import Path
from transformers import AutoConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, timeout=7200):
    """Run command with enhanced error handling, logging, and real-time output"""
    logger.info(f"Executing command: {command}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        stdout_lines = []
        stderr_lines = []
        
        # Function to handle output streams
        def handle_output(pipe, lines):
            for line in iter(pipe.readline, ''):
                print(line, end='')  # Print in real-time
                lines.append(line)
            pipe.close()
        
        # Create threads to handle stdout and stderr
        import threading
        stdout_thread = threading.Thread(target=handle_output, args=(process.stdout, stdout_lines))
        stderr_thread = threading.Thread(target=handle_output, args=(process.stderr, stderr_lines))
        
        # Start threads
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        process.wait(timeout=timeout)
        
        # Wait for output threads to complete
        stdout_thread.join()
        stderr_thread.join()
        
        duration = time.time() - start_time
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)
        
        logger.info(f"Command completed in {duration:.2f} seconds")
        
        # Immediately fail if there's any stderr output or non-zero return code
        if stderr:
            logger.error(f"Command stderr: {stderr}")
            raise RuntimeError(f"Command produced stderr output: {stderr}")
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            logger.error(f"stdout: {stdout}")
            logger.error(f"stderr: {stderr}")
            raise subprocess.CalledProcessError(process.returncode, command)
            
        return stdout, stderr
        
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
        try:
            if filepath.exists():
                size = filepath.stat().st_size
                if size > 0:
                    # Add a small delay to ensure file is fully written
                    time.sleep(5)
                    # Verify size hasn't changed
                    new_size = filepath.stat().st_size
                    if new_size == size:
                        logger.info(f"File {filepath} exists and is stable (size: {size} bytes)")
                        return True
                    else:
                        logger.warning(f"File {filepath} size changed from {size} to {new_size}, still writing...")
                else:
                    logger.warning(f"File {filepath} exists but is empty (size: {size} bytes)")
            else:
                logger.debug(f"File {filepath} does not exist yet, waiting...")
        except (OSError, IOError) as e:
            logger.warning(f"Error checking file: {str(e)}")
        time.sleep(check_interval)
    
    logger.error(f"Timeout waiting for file {filepath} after {timeout} seconds")
    return False

def get_results(results_file):
    """Get results with enhanced error handling"""
    try:
        if not verify_file_exists(results_file):
            raise FileNotFoundError(f"Results file {results_file} not found or empty")
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Validate results structure
        required_keys = ['results', 'wmdp_bio', 'mmlu']
        for key in required_keys:
            if key not in results.get('results', {}):
                raise KeyError(f"Missing required key {key} in results")
        
        return {
            'wmdp_bio': results['results']['wmdp_bio']['acc,none'],
            'mmlu': results['results']['mmlu']['acc']
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse results file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise

def get_module_params(param_type):
    """Return parameter indices based on type"""
    param_indices = {
        "mlp": "4,5,6,7",
        "attention": "0,1,2,3",
        "both": "0,1,2,3,4,5,6,7"
    }
    
    if param_type not in param_indices:
        logger.error(f"Invalid param_type: {param_type}")
        raise ValueError(f"Invalid param_type: {param_type}")
        
    return param_indices[param_type]

def objective(trial):
    trial_start_time = time.time()
    logger.info(f"\nStarting trial {trial.number}")
    
    try:
        # Simplified hyperparameter search space
        params = {
            'layer_id': trial.suggest_int('layer_id', 0, 15),
            'num_batches': 150,  # Fixed value
            'alpha': trial.suggest_float('alpha', 300.0, 2000.0),
            'steering_coeff': trial.suggest_float('steering_coeff', 1.0, 300.0, log=True),
            'lr': trial.suggest_float('lr', 1e-5, 1e-4, log=True),
            'batch_size': 16,  # Fixed value
            'module_type': 'mlp',  # Fixed value
            'window_size': trial.suggest_int('window_size', 1, 3),
        }
        
        logger.info(f"Trial {trial.number} parameters: {json.dumps(params, indent=2)}")
        
        # Create trial directory and define model file path
        trial_dir = Path(f"models/llama_opt_trial_{trial.number}")
        trial_dir.mkdir(parents=True, exist_ok=True)
        model_file = trial_dir / "pytorch_model.bin"
        
        # Save trial parameters
        with open(trial_dir / "params.json", 'w') as f:
            json.dump(params, f, indent=2)
        
        # Construct layer IDs with bounds checking
        if params['window_size'] == 1:
            layer_ids = str(params['layer_id'])
        else:
            start_layer = max(0, params['layer_id'] - params['window_size'] + 1)
            end_layer = min(15, params['layer_id'] + 1)
            layers = range(start_layer, end_layer)
            layer_ids = ','.join(map(str, layers))
        
        # Get parameter indices
        param_ids = get_module_params(params['module_type'])
        
        # Construct and run unlearn command
        unlearn_command = (
            f"python3 -m old_rmu.unlearn "
            f"--max_num_batches {params['num_batches']} "
            f"--batch_size={params['batch_size']} "
            f"--retain_corpora wikitext "
            f"--forget_corpora bio-forget-corpus "
            f"--steering_coeffs {params['steering_coeff']} "
            f"--alpha {params['alpha']} "
            f"--lr {params['lr']} "
            f"--seed 42 "
            f"--output_dir {trial_dir} "
            f"--model meta-llama/Llama-3.2-1B-Instruct "
            f"--device cuda "
            f"--layer_id {params['layer_id']} "
            f"--layer_ids {layer_ids} "
            f"--param_ids {param_ids}"
        )
        
        # Run unlearning - will raise exception on any error
        stdout, stderr = run_command(unlearn_command)
        
        # Verify model file exists and is valid
        if not verify_file_exists(model_file, timeout=900):  # 15 minutes timeout
            raise FileNotFoundError(f"Model file {model_file} not created or empty after unlearning")
        
        # Run evaluation
        eval_command = (
            f"lm-eval --model hf "
            f"--model_args pretrained={trial_dir} "
            f"--tasks wmdp,mmlu "
            f"--batch_size=16 "
            f"--output_path {trial_dir}/results.json"
        )
        
        # Run evaluation - will raise exception on any error
        stdout, stderr = run_command(eval_command)
        
        # Get and process results
        results = get_results(f"{trial_dir}/results.json")
        
        # Calculate objective
        baseline_mmlu = 0.45
        mmlu_penalty = max(0, baseline_mmlu - results['mmlu']) * 2.0
        objective_value = results['wmdp_bio'] + mmlu_penalty
        
        # Store metrics
        trial.set_user_attr('wmdp_bio_acc', results['wmdp_bio'])
        trial.set_user_attr('mmlu_acc', results['mmlu'])
        trial.set_user_attr('mmlu_penalty', mmlu_penalty)
        
        # Log trial completion
        duration = time.time() - trial_start_time
        logger.info(f"Trial {trial.number} completed in {duration:.2f} seconds")
        logger.info(f"Trial {trial.number} objective value: {objective_value}")
        
        return objective_value
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
        # Clean up trial directory on failure
        if trial_dir.exists():
            import shutil
            shutil.rmtree(trial_dir)
        # Re-raise the exception to stop the trial
        raise

def main():
    study_name = "llama_unlearning_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    # Add custom pruner to stop unpromising trials early
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
    
    study.optimize(objective, n_trials=100)
    
    # Save and visualize results
    save_results(study)

def save_results(study):
    os.makedirs('optimization_results', exist_ok=True)
    
    # Basic statistics
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("  Metrics:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")
    
    # Save detailed results
    study_stats = {
        'best_trial': {
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs
        },
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'user_attrs': t.user_attrs
            }
            for t in study.trials if t.value != float('inf')
        ]
    }
    
    with open('optimization_results/study_results.json', 'w') as f:
        json.dump(study_stats, f, indent=2)
    
    # Create visualizations
    create_visualizations(study)

def create_visualizations(study):
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