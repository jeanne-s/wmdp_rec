# The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning 

Replication of the [WMDP paper](https://www.wmdp.ai/).

# RMU Script 
This code implements the RMU algorithm for model unlearning.

Basic usage:
```bash
python src/rmu.py --config_file yaml_files/finetuning/your_config.yaml
```

### YAML Configuration
Create a YAML configuration file with the following parameters:

```yaml
args:
  # Model Configuration
  model_name: "HuggingFaceH4/zephyr-7b-beta"    # HuggingFace model name/path
  updated_model_path: "models/"                  # Output directory for the modified model

  # Training Parameters
  num_epochs: 1                                  # Number of training epochs
  num_batches: 150                              # Number of batches per epoch
  batch_size: 4                                 # Batch size for training
  learning_rate: 5e-5                          # Learning rate for optimizer
  
  # Dataset Configuration
  forget_dataset_list:                         # List of datasets to unlearn
    - "cyber-forget-corpus.jsonl"
    - "bio-forget-corpus.jsonl"
  retain_dataset_list:                         # List of datasets to retain
    - "wikitext"

  # RMU-specific Parameters
  steering_coefficient: 6.5                    # Coefficient for steering vector
  alpha: 1200.0                               # Weight for retain loss
  forget_layer_id: 7                          # Layer ID for forget operations
  
  # Layer Configuration
  optimizer_param_layer_id:                    # Layer IDs for optimizer parameters
    - 6
  update_layer_ids:                           # Layer IDs to update during training
    - 5
    - 6
    - 7
```

# Benchmarking Tool 
This code allows benchmarking and comparison of performance of the base and unlearned models across various tasks including WMDP and MMLU. 

Basic Usage:
```bash 
python src/benchmark.py --config_file yaml_files/benchmarking/your_config.yaml
```

### Command Line Arguments

- `--config_file` (required): Path to the YAML configuration file
  ```bash
  python src/benchmark.py --config_file yaml_files/benchmarking/zephyr-7b.yaml
  ```

- `--device` (optional): Device to use for evaluation (default: "cuda")
  ```bash
  python src/benchmark.py --config_file config.yaml --device cpu
  ```

- `--plot_only` (optional): Only generate plots without running evaluations
  ```bash
  python src/benchmark.py --config_file config.yaml --plot_only
  ```

- `--rerun_base` (optional): Force re-run of base model evaluation even if results exist
  ```bash
  python src/benchmark.py --config_file config.yaml --rerun_base
  ```

### Plots Generated

1. `accuracy_comparison.png`: Compares base and unlearned model performance across WMDP and MMLU tasks
2. `mmlu_subcategories.png`: Detailed comparison across MMLU subcategories