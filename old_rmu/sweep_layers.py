import subprocess
import json
import os

# Llama 3.2 1B model configuration
from transformers import AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')

def get_wmdp_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results['results']['wmdp']['acc']

# Create a directory to store all results
os.makedirs('sweep_results', exist_ok=True)

results = []

for layer_id in range(1, num_layers - 1):  # Skip first and last layers
    layer_ids = f"{layer_id-1},{layer_id},{layer_id+1}"
    param_ids = ','.join(map(str, range(num_heads)))
    
    # Run unlearn.py
    unlearn_command = f"python3 -m old_rmu.unlearn --max_num_batches 150 --batch_size=4 --retain_corpora wikitext --forget_corpora bio-forget-corpus --steering_coeffs 6.5 --alpha 1200 --lr 5e-5 --seed 42 --output_dir models/llama_rmu_{layer_id} --model meta-llama/Llama-3.2-1B-Instruct --device cuda --layer_id {layer_id} --layer_ids {layer_ids} --param_ids {param_ids}"
    print(f"Running: {unlearn_command}")
    stdout, stderr = run_command(unlearn_command)
    
    # Run lm-eval
    eval_command = f"lm-eval --model hf --model_args pretrained=models/llama_rmu_{layer_id} --tasks wmdp --batch_size=16 --output_path models/llama_rmu_{layer_id}/results.json"
    print(f"Running: {eval_command}")
    stdout, stderr = run_command(eval_command)
    
    # Get WMDP results
    wmdp_acc = get_wmdp_results(f"models/llama_rmu_{layer_id}/results.json")
    
    results.append({
        'layer_id': layer_id,
        'layer_ids': layer_ids,
        'param_ids': param_ids,
        'wmdp_acc': wmdp_acc
    })
    
    # Save intermediate results
    with open('sweep_results/intermediate_results.json', 'w') as f:
        json.dump(results, f, indent=2)

# Save final results
with open('sweep_results/final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plotting code
import matplotlib.pyplot as plt
import numpy as np

# Create line plot
plt.figure(figsize=(12, 8))

layer_ids = [r['layer_id'] for r in results]
accuracies = [r['wmdp_acc'] for r in results]

plt.plot(layer_ids, accuracies, marker='o')
plt.xlabel('Layer ID')
plt.ylabel('WMDP Accuracy')
plt.title('WMDP Accuracy vs Layer ID')
plt.grid(True)

# Save the plot
plt.savefig('sweep_results/accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Parameter sweep completed. Results and plot saved in sweep_results/")