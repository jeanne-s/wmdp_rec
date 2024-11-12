import argparse
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
import tqdm as tqdm
from dataset import JSONLDataset, WikitextDataset
from model import Model
from utils import load_yaml_config
import copy
from dotenv import load_dotenv
load_dotenv()
from logger_config import setup_logger
from typing import List, Optional, Union
import logging
from exceptions import ModelLoadError, ConfigurationError

logger = logging.getLogger(__name__)


class BaseRMU: 
    
    def __init__(self, args):
        self.args = args
    

    def setup(self):
        """
        This method sets up the tokenizer and optimizer for the model.
        """
        SEED = self.args.seed
        torch.manual_seed(SEED)
        device = getattr(self.args, 'device', 'cuda')
        if device == 'cuda':
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                logger.warning("CUDA is not available. Using CPU instead.")
            self.device = torch.device(device if cuda_available else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        self.torch_dtype = torch.bfloat16
        self.tokenizer = self._load_tokenizer()
        self.updated_model, self.frozen_model = self._load_models()
        self.optimizer = self._load_optimizer() 
        self.control_vector_list = self._create_control_vector_list()
        self.retain_datasets, self.forget_datasets = self._setup_datasets()
        

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, 
                                            trust_remote_code=True, 
                                            use_fast=False)
        
        # Set padding token to be the same as EOS token if pad token is not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer 


    def _load_models(self):
        try:
            # Load a single instance of the model
            updated_model = Model(model_name=self.args.model_name,
                                torch_dtype=self.torch_dtype,
                                device=self.device)

            # Initially freeze all parameters
            for param in updated_model.model.parameters():
                param.requires_grad = False

            frozen_model = copy.deepcopy(updated_model)

            # Validate layer IDs
            max_layers = updated_model.n_layers()
            for layer_id in self.args.update_layer_ids:
                if not 0 <= layer_id < max_layers:
                    raise ConfigurationError(
                        f"Invalid layer_id {layer_id}. Must be between 0 and {max_layers-1}"
                    )
                for param in updated_model.get_layer(layer_id).parameters():
                    param.requires_grad = True

            return updated_model, frozen_model
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load models: {str(e)}") from e


    def _load_optimizer(self):
        optimizer_param_layer_id = set(self.args.optimizer_param_layer_id)
        params = [
            p
            for layer_id in self.args.update_layer_ids
            if 0 <= layer_id < self.updated_model.n_layers()
            for i, p in enumerate(self.updated_model.get_layer(layer_id).parameters())
            if i in optimizer_param_layer_id
        ]
        optimizer = AdamW(params, lr=float(self.args.learning_rate)) 
        return optimizer
    

    def _create_control_vector_list(self):
        """ 
        Samples a unit vector with independent entries drawn uniformly at random from [0,1).
        Corresponds to u in the paper. One unit vector is created per forget dataset; 
        for each forget dataset u is held fixed throughout training.
        """
        control_vector_list = []
        for _ in range(len(self.args.forget_dataset_list)):
            
            control_vector = torch.rand(1, 1, self.updated_model.model.config.hidden_size,
                                        device=self.device)
            normalized_control_vector = control_vector / torch.norm(control_vector)
            control_vector_list.append(normalized_control_vector)
        return control_vector_list

    
    def _setup_datasets(self):
        retain_datasets = []
        for dataset_id, retain_dataset in enumerate(self.args.retain_dataset_list):
            if retain_dataset.endswith('.jsonl'):
                retain_datasets.append(
                    JSONLDataset(dataset_name=retain_dataset,
                                 tokenizer=self.tokenizer,
                                 batch_size=self.args.batch_size)
                )
            else: # The retain dataset is wikitext
                retain_datasets.append(
                    WikitextDataset(tokenizer=self.tokenizer,
                                    batch_size=self.args.batch_size)
                )

        forget_datasets = []
        for dataset_id, forget_dataset_name in enumerate(self.args.forget_dataset_list):
            forget_datasets.append(
                JSONLDataset(dataset_name=forget_dataset_name,
                             tokenizer=self.tokenizer,
                             batch_size=self.args.batch_size)
            )

        # If the length of retain_datasets is smaller than forget_datasets, we have to extend it for the finetuning loop to function
        if len(retain_datasets) < len(forget_datasets):
            retain_datasets *= (len(forget_datasets) // len(retain_datasets))  # Repeat the list
            retain_datasets += retain_datasets[:len(forget_datasets) % len(retain_datasets)]  # Add remaining items if needed
        
        assert len(retain_datasets) == len(forget_datasets)
        return retain_datasets, forget_datasets

    
    def forget_loss(self, 
                    x_forget, 
                    control_vector):
        """Calculates the forget loss.
        Args:
            x_forget (torch.Tensor): The input tokens.
        """ 
        # Ensure inputs are on the correct device
        x_forget = x_forget.to(self.device)
        control_vector = control_vector.to(self.device)
        
        updated_model_activations = self.updated_model.forward(
            input_ids=x_forget, 
            layer_id=self.args.forget_layer_id,
            no_grad=False
        )
        
        # Convert to float32 only when using CPU
        if self.device.type == 'cpu':
            updated_model_activations = updated_model_activations.to(dtype=torch.float32)
            control_vector = control_vector.to(dtype=torch.float32)
        
        # Ensure the control_vector is broadcastable
        control_vector = control_vector.expand_as(updated_model_activations)

        return torch.nn.functional.mse_loss(
            updated_model_activations, 
            control_vector * self.args.steering_coefficient
        )
    

    def retain_loss(self, 
                    x_retain):
        """Calculates the retain loss. 
        Args:
            x_retain (torch.Tensor): The input tokens.
        """
        # Ensure input is on the correct device
        x_retain = x_retain.to(self.device)
        
        updated_model_activations = self.updated_model.forward(
            input_ids=x_retain, 
            layer_id=self.args.forget_layer_id,
            no_grad=False
        )
        
        frozen_model_activations = self.frozen_model.forward(
            input_ids=x_retain,
            layer_id=self.args.forget_layer_id, 
            no_grad=True
        )
        
        # Convert to float32 only when using CPU
        if self.device.type == 'cpu':
            updated_model_activations = updated_model_activations.to(dtype=torch.float32)
            frozen_model_activations = frozen_model_activations.to(dtype=torch.float32)

        return torch.nn.functional.mse_loss(updated_model_activations, frozen_model_activations)


    def finetune(self):
        """Main training loop."""
        
        for epoch in range(self.args.num_epochs):
            with tqdm.tqdm(total=self.args.num_batches) as pbar:
                for batch_id in range(self.args.num_batches):
                
                    # To unlearn multiple datasets, we interleave the gradient updates
                    # i.e. we update model weights on the first forget dataset, then on 
                    # the second one, and repeat
                    dataset_id = batch_id % len(self.forget_datasets)
                    element_id = batch_id // len(self.forget_datasets)
                    
                    x_forget = self.forget_datasets[dataset_id][element_id]['input_ids']
                    x_retain = self.retain_datasets[dataset_id][element_id]['input_ids']
                    control_vector = self.control_vector_list[dataset_id].to(self.device, dtype=self.torch_dtype)
        
                    l_forget = self.forget_loss(x_forget=x_forget, control_vector=control_vector)
                    l_retain = self.args.alpha * self.retain_loss(x_retain=x_retain)
                    full_loss = l_forget + l_retain

                    self.optimizer.zero_grad()
                    full_loss.backward()
                    self.optimizer.step()
                    print(f"Step {batch_id}/{self.args.num_batches}: full_loss={full_loss.item():.5g}, forget_loss={l_forget.item():.5g}, retain_loss={l_retain.item():.5g}")

        self.updated_model.save_model(path=self.args.updated_model_path, 
                                      config_path=self.args.config_file)
        return    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMU Training")
    parser.add_argument('--config_file', type=str, help='Path to the YAML config file')
    args = parser.parse_args()
    
    config = load_yaml_config(file_path=args.config_file)
    setattr(config, 'config_file', args.config_file)
    
    rmu = BaseRMU(config)
    rmu.setup()
    rmu.finetune()