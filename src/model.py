import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import yaml
import shutil
from dotenv import load_dotenv
load_dotenv()


class Model:

    def __init__(self, 
                 model_name: str,
                 torch_dtype,
                 device=None
    ):
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()


    def load_model(self):
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=self.torch_dtype,
            use_auth_token=hf_token
        )
        # Move the entire model to the specified device
        return model.to(self.device)

    
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
        # Ensure input_ids is on the correct device
        input_ids = input_ids.to(self.device)
        
        activations = []
        def hook_function(module, input, output):
            # Ensure output is on the correct device
            act = output[0] if isinstance(output, tuple) else output
            activations.append(act.to(self.device))

        hook_handle = self.get_layer(layer_id).register_forward_hook(hook_function)
        
        if no_grad:
            with torch.no_grad():
                _ = self.model(input_ids)
        else:
            _ = self.model(input_ids)

        hook_handle.remove()
        activations_tensor = torch.stack(activations)
        return activations_tensor.to(self.device)  # Ensure final output is on correct device
    

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