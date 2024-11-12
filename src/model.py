import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import os
import yaml
import shutil
from dotenv import load_dotenv
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

load_dotenv()


class Model:
    """
    Wrapper class for managing language models.

    This class provides a unified interface for loading, saving, and accessing
    language models and their tokenizers. It handles device placement and 
    configuration management.

    Attributes:
        model_name (str): Name or path of the model
        device (str): Device to place the model on ('cuda' or 'cpu')
        torch_dtype: Data type for model parameters
        model (PreTrainedModel): Loaded model instance
        tokenizer (PreTrainedTokenizer): Model's tokenizer
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False
    ) -> None:
        """
        Initialize the model wrapper.

        Args:
            model_name: HuggingFace model name or path
            device: Device to place model on
            torch_dtype: Optional specific torch dtype for model
            trust_remote_code: Whether to trust remote code in model loading

        Raises:
            ValueError: If device is 'cuda' but CUDA is not available
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype or torch.float16
        self.trust_remote_code = trust_remote_code
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self) -> PreTrainedModel:
        """
        Load model from HuggingFace hub with proper authentication and configuration.

        Returns:
            Loaded model instance placed on the specified device
        """
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=self.torch_dtype,
            use_auth_token=hf_token
        )
        # Move the entire model to the specified device
        return model.to(self.device)

    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and configure the appropriate tokenizer for the model.

        Returns:
            Configured tokenizer instance
        """
        return AutoTokenizer.from_pretrained(self.model_name)


    def get_all_layers(self) -> List[torch.nn.Module]:
        if hasattr(self.model, 'transformer'):  # For models like GPT-2
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):  # For other models (e.g., zephyr, Yi, Mixtral)
            layers = self.model.model.layers
        else:
            raise ValueError(f"Unknown architecture for model {self.model_name}. Unable to find layers.")
        return layers


    def get_layer(self, layer_id: int) -> torch.nn.Module:
        """
        Get a specific layer from the model.

        Args:
            layer_id: Index of the desired layer

        Returns:
            The specified model layer

        Raises:
            IndexError: If layer_id is out of range
        """
        layers = self.get_all_layers()
        if 0 <= layer_id < len(layers):
            return layers[layer_id]
        else:
            raise IndexError(f"Layer ID {layer_id} is out of range. The model has {len(layers)} layers.")

        
    def n_layers(self) -> int:
        """Returns the number of layers in the model."""
        layers = self.get_all_layers()
        return len(layers)


    def forward(self, 
                input_ids, 
                layer_id: int,
                no_grad=True
    ) -> torch.Tensor:
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
        return activations_tensor.to(self.device)  # Ensure final output is on the correct device
    

    def save_model(
        self, 
        path: Union[str, Path], 
        config_path: str = None,
        save_tokenizer: bool = True
    ) -> None:
        """
        Save the model and optionally its tokenizer.

        Args:
            path: Directory path to save the model
            save_tokenizer: Whether to also save the tokenizer

        Raises:
            OSError: If saving fails due to permissions or disk space
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