import json
import glob
import os
import matplotlib.pyplot as plt

def load_model_results(base_dir='models'):
    """Load results from individual model result files."""
    results = {
        'wmdp': [],
        'wmdp_bio': [],
        'wmdp_chem': [],
        'wmdp_cyber': []
    }
    
    pattern = os.path.join(base_dir, 'llama_rmu_*', 'results.json')
    
    for filepath in sorted(glob.glob(pattern)):
        # Extract layer_id from directory name
        layer_id = int(filepath.split('llama_rmu_')[-1].split('/')[0])
        
        # Load results
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # Store results for each category
            for category in results.keys():
                acc = data['results'][category]['acc,none']
                results[category].append({
                    'layer_id': layer_id,
                    'acc': acc
                })
    
    # Sort results by layer_id for each category
    for category in results:
        results[category].sort(key=lambda x: x['layer_id'])
    
    return results

def create_accuracy_plot(results, output_path):
    """Create and save a line plot of WMDP accuracies vs layer ID."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'wmdp': 'blue',
        'wmdp_bio': 'green',
        'wmdp_chem': 'red',
        'wmdp_cyber': 'purple'
    }
    
    labels = {
        'wmdp': 'WMDP (Overall)',
        'wmdp_bio': 'Biology',
        'wmdp_chem': 'Chemistry',
        'wmdp_cyber': 'Cybersecurity'
    }
    
    for category in results:
        layer_ids = [r['layer_id'] for r in results[category]]
        accuracies = [r['acc'] for r in results[category]]
        
        plt.plot(layer_ids, accuracies, marker='o', linewidth=2, markersize=6, 
                label=labels[category], color=colors[category])
    
    plt.xlabel('Layer ID', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('WMDP Accuracy by Category vs Layer ID', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add minor ticks and grid
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    results = load_model_results()
    
    if not any(results.values()):
        print("No results found! Please check models/llama_rmu_*/results.json exist.")
        return
    
    # Create and save the plot
    create_accuracy_plot(results, 'sweep_results/accuracy_plot.png')
    print(f"Plot saved as sweep_results/accuracy_plot.png (using {len(results['wmdp'])} layer results)")

if __name__ == "__main__":
    main()