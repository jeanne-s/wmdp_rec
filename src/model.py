import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import yaml
import shutil

class Model:

    def __init__(self, 
                 model_name: str,
                 torch_dtype
    ):
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def load_model(self):
        if torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map='auto',
                torch_dtype=self.torch_dtype
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='cpu',
                torch_dtype=self.torch_dtype
            )

    
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)


    def get_all_layers(self):
        if hasattr(self.model, 'transformer'):  # For models like GPT-2
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):  # For other models (e.g., zephyr, Yi, Mixtral)
            layers = self.model.model.layers
        else:
            raise ValueError(f"Unknown architecture for model {self.model_name}. Unable to find layers.")
        return layers


    def get_layer(self, layer_id):
        """Returns the layer object given its layer index (layer_id)."""
        layers = self.get_all_layers()
        if 0 <= layer_id < len(layers):
            return layers[layer_id]
        else:
            raise IndexError(f"Layer ID {layer_id} is out of range. The model has {len(layers)} layers.")

        
    def n_layers(self):
        """Returns the number of layers in the model."""
        layers = self.get_all_layers()
        return len(layers)


    def forward(self, 
                input_ids, 
                layer_id: int,
                no_grad=True
    ):
        """Forward pass and returns the activations of the specified layer."""
        activations = []
        def hook_function(module, input, output):
            activations.append(output[0].to(self.device) if isinstance(output, tuple) else output.to(self.device))

        hook_handle = self.get_layer(layer_id).register_forward_hook(hook_function)
        
        if no_grad:
            with torch.no_grad():
                _ = self.model(input_ids)
        else:
            _ = self.model(input_ids)

        hook_handle.remove()
        activations_tensor = torch.stack(activations)
        return activations_tensor
    

    def save_model(self, 
                   path: str,
                   config_path: str = None
    ):
        """
        Saves the model to the specified path. If a folder with the model's name already exists,
        it creates a subfolder with an incremental number (e.g., 00, 01, etc.). The model is saved
        in the subfolder along with a configuration file in YAML format.

        Args:
            path (str): The base path where the model will be saved.
            config_path (str): The path to the configuration file (YAML) used to run rmu.py.
        """
        model_dir = os.path.join(path, self.model_name)

        # If the folder exists, create subfolders with incremental numbers (e.g., 00, 01, etc.)
        if os.path.exists(model_dir):
            subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
            numbers = [int(f) for f in subfolders if f.isdigit()]
            new_subfolder_num = f"{max(numbers) + 1:02}" if numbers else "00"
        else:
            # If folder doesn't exist, create the main folder and start with subfolder 00
            os.makedirs(model_dir)
            new_subfolder_num = "00"

        # Create the subfolder path
        save_path = os.path.join(model_dir, new_subfolder_num)
        os.makedirs(save_path, exist_ok=True)

        # Save the model
        self.model.save_pretrained(os.path.join(save_path, "model.pt"))

        # Save the tokenizer (needed to run benchmarks)
        self.tokenizer.save_pretrained(os.path.join(save_path, "model.pt"))

        # Copy the external config file (used in run.py) to the save directory, if provided
        if config_path:
            config_destination = os.path.join(save_path, "config.yaml")
            shutil.copy(config_path, config_destination)
        print(f'Model saved at {save_path}.')
        return