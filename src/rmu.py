import argparse
import torch
from transformers import AdamW, AutoTokenizer
import tqdm as tqdm
from dataset import JSONLDataset
from model import Model
from utils import load_yaml_config


class BaseRMU: 
    
    def __init__(self, args):
        self.args = args
    

    def setup(self):
        """
        This method sets up the tokenizer and optimizer for the model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self.load_tokenizer()
        self.updated_model, self.frozen_model = self.load_models()
        self.optimizer = self.load_optimizer() 
        self.control_vector_list = self.create_control_vector_list()
        self.retain_datasets, self.forget_datasets = self.setup_datasets()
        

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, 
                                                  trust_remote_code_true=True, 
                                                  use_fast=False)
        return tokenizer 

    
    # def load_models(self):
    #     updated_model = Model(model_name=self.args.model_name)
    #     # updated_model.model.to(self.device)
    #     frozen_model = Model(model_name=self.args.model_name)
    #     # frozen_model.model.to(self.device)
    #     return updated_model, frozen_model

    def load_models(self):
        updated_model = Model(model_name=self.args.model_name)#.model.to(self.device)
        frozen_model = updated_model  # Use the same model instance for both
        frozen_model.model.eval()  # Set the frozen model to evaluation mode
        for param in frozen_model.model.model.parameters():
            param.requires_grad = False  # Freeze the parameters of the frozen model
        return updated_model, frozen_model


    def load_optimizer(self):
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
    

    def create_control_vector_list(self):
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

    
    def setup_datasets(self):
        retain_datasets = []
        for dataset_id, retain_dataset in enumerate(self.args.retain_dataset_list):
            retain_datasets.append(
                JSONLDataset(dataset_name=retain_dataset,
                             tokenizer=self.tokenizer)
            )

        forget_datasets = []
        for dataset_id, forget_dataset_name in enumerate(self.args.forget_dataset_list):
            forget_datasets.append(
                JSONLDataset(dataset_name=forget_dataset_name,
                             tokenizer=self.tokenizer)
            )

        # If the length of retain_datasets is smaller than forget_datasets, we have to extend it for the 
        # finetuning loop to function
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
        updated_model_activations = self.updated_model.forward(input_ids=x_forget, 
                                                               layer_id=self.args.forget_layer_id,
                                                               no_grad=False)
        L_f = x_forget.shape[0]

        # Check the shape of updated_model_activations
        if isinstance(updated_model_activations, list):
            updated_model_activations = torch.stack(updated_model_activations)
        elif isinstance(updated_model_activations, torch.Tensor):
            if updated_model_activations.dim() == 0:
                updated_model_activations = updated_model_activations.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected type for updated_model_activations: {type(updated_model_activations)}")

        # Convert updated_model_activations to a tensor if it's a list
        updated_model_activations = torch.tensor(updated_model_activations) if isinstance(updated_model_activations, list) else updated_model_activations
        
        return 1/L_f * torch.nn.functional.mse_loss(updated_model_activations, control_vector * self.args.steering_coefficient)
    

    def retain_loss(self, 
                    x_retain):
        """Calculates the retain loss. 
        Args:
            x_retain (torch.Tensor): The input tokens.
        """
        updated_model_activations = self.updated_model.forward(input_ids=x_retain, 
                                                               layer_id=self.args.forget_layer_id,
                                                               no_grad=False)
        frozen_model_activations = self.frozen_model.forward(input_ids=x_retain,
                                                             layer_id=self.args.forget_layer_id, 
                                                             no_grad=True)
        L_r = x_retain.shape[0]
        
        # Convert frozen_model_activations to a tensor if it's a list
        if isinstance(frozen_model_activations, list):
            frozen_model_activations = torch.stack(frozen_model_activations)

        # Ensure updated_model_activations is also a tensor
        if isinstance(updated_model_activations, list):
            updated_model_activations = torch.stack(updated_model_activations)

        return 1/L_r * torch.nn.functional.mse_loss(updated_model_activations, frozen_model_activations)


    def finetune(self):
        """Main training loop."""
        # Add this block before tokenizing or creating datasets
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.model.config.pad_token_id = self.model.config.eos_token_id
        
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
                    control_vector = self.control_vector_list[dataset_id]
            
                    l_forget = self.forget_loss(x_forget=x_forget, control_vector=control_vector)
                    l_retain = self.retain_loss(x_retain=x_retain)
                    full_loss = l_forget + self.args.alpha * l_retain

                    self.optimizer.zero_grad()
                    # Add this before full_loss.backward()
                    # for name, param in model.named_parameters():
                    #     if not param.requires_grad:
                    #         print(f"Parameter {name} does not require gradients")

                    # Then call backward
                    full_loss.backward()
                    self.optimizer.step()

                    print(f"Step {batch_id}: loss={full_loss.item():.4f}, forget_loss={l_forget.item():.4f}, retain_loss={l_retain.item():.4f}")

        self.updated_model.save_model(path=self.args.updated_model_path, config_path=self.args.config_file)
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