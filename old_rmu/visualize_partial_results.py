import optuna
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

def load_existing_trials():
    """Load completed trials from existing results files"""
    trials = []
    models_dir = Path("models")
    
    # Find all trial directories
    for trial_dir in models_dir.glob("llama_opt_trial_*"):
        try:
            # Load parameters
            with open(trial_dir / "params.json", "r") as f:
                params = json.load(f)
            
            # Load results
            with open(trial_dir / "results.json", "r") as f:
                results = json.load(f)
                
            # Calculate objective value (same as in original script)
            baseline_mmlu = 0.45
            mmlu_score = results.get('mmlu', {}).get('acc,none', 0.0)
            wmdp_bio_score = results.get('wmdp_bio', {}).get('acc,none', 0.0)
            mmlu_penalty = max(0, baseline_mmlu - mmlu_score) * 2.0
            objective_value = wmdp_bio_score + mmlu_penalty
            
            # Create trial-like object
            trial_info = {
                'number': int(trial_dir.name.split('_')[-1]),
                'value': objective_value,
                'params': params,
                'user_attrs': {
                    'wmdp_bio_acc': wmdp_bio_score,
                    'mmlu_acc': mmlu_score,
                    'mmlu_penalty': mmlu_penalty
                }
            }
            trials.append(trial_info)
            
        except Exception as e:
            print(f"Error processing {trial_dir}: {e}")
            continue
    
    return trials
def create_visualizations(trials):
    """Create visualization plots from trial data"""
    os.makedirs('partial_optimization_results', exist_ok=True)
    
    # Save trial data
    with open('partial_optimization_results/current_results.json', 'w') as f:
        json.dump(trials, f, indent=2)
    
    # Find best trial
    best_trial = min(trials, key=lambda x: x['value'])
    print("\nBest trial so far:")
    print(f"  Value: {best_trial['value']}")
    print("  Params: ")
    for key, value in best_trial['params'].items():
        print(f"    {key}: {value}")
    print("  Metrics:")
    for key, value in best_trial['user_attrs'].items():
        print(f"    {key}: {value}")
    
    # Create parameter importance visualization
    param_values = {param: [] for param in trials[0]['params'].keys()}
    objective_values = []
    
    for trial in trials:
        for param, value in trial['params'].items():
            param_values[param].append(value)
        objective_values.append(trial['value'])
    
    # Parameter correlation plot
    for param in param_values:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=param_values[param],
            y=objective_values,
            mode='markers',
            name=param
        ))
        fig.update_layout(
            title=f"{param} vs Objective Value",
            xaxis_title=param,
            yaxis_title="Objective Value"
        )
        pio.write_image(fig, f"partial_optimization_results/{param}_correlation.png")
    
    # Optimization history
    trial_numbers = [t['number'] for t in trials]
    values = [t['value'] for t in trials]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=values,
        mode='lines+markers',
        name='Objective Value'
    ))
    fig.update_layout(
        title="Optimization History",
        xaxis_title="Trial Number",
        yaxis_title="Objective Value"
    )
    pio.write_image(fig, "partial_optimization_results/optimization_history.png")
    
    # Parallel coordinates plot - only include numerical parameters
    numerical_params = {}
    for param in param_values:
        # Check if all values are numerical
        if all(isinstance(x, (int, float)) for x in param_values[param]):
            numerical_params[param] = param_values[param]
    
    dimensions = [
        dict(range=[min(numerical_params[param]), max(numerical_params[param])],
             label=param,
             values=numerical_params[param])
        for param in numerical_params.keys()
    ]
    dimensions.append(
        dict(range=[min(objective_values), max(objective_values)],
             label='Objective Value',
             values=objective_values)
    )
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=objective_values,
                 colorscale='Viridis'),
        dimensions=dimensions
    ))
    fig.update_layout(title="Parallel Coordinates Plot")
    pio.write_image(fig, "partial_optimization_results/parallel_coordinates.png")

def main():
    print("Loading existing trials...")
    trials = load_existing_trials()
    print(f"Found {len(trials)} completed trials")
    
    print("Creating visualizations...")
    create_visualizations(trials)
    print("\nVisualizations have been saved to the 'partial_optimization_results' directory")

if __name__ == "__main__":
    main()