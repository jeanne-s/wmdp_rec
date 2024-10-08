import torch
from transformers import AutoModelForCausalLM


class Model:

    def __init__(self, 
                 model_name: str
    ):
        self.model_name = model_name
        self.model = self.load_model()


    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name)


    def get_all_layers(self):
        # TODO: dictionary
        if hasattr(self.model, 'transformer'):  # For models like GPT-2
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):  # For specific models (e.g., zephyr, Mixtral)
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
            activations.append(output[0] if isinstance(output, tuple) else output)

        hook_handle = self.get_layer(layer_id).register_forward_hook(hook_function)
        
        if no_grad:
            with torch.no_grad():
                _ = self.model(**input_ids)
        else:
            _ = self.model(**input_ids)

        hook_handle.remove()
        return activations
    

    def save_model(self, 
                   path: str
    ):
        self.model.save_pretrained(path)
        return path