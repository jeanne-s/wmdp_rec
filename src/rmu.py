import argparse
import torch
from transformers import AdamW, AutoTokenizer
import tqdm as tqdm
from dataset import JSONLDataset, WikitextDataset
from model import Model
from utils import load_yaml_config
import copy


class BaseRMU: 
    
    def __init__(self, args):
        self.args = args
    

    def setup(self):
        """
        This method sets up the tokenizer and optimizer for the model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16
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


    def load_models(self):
        # Load a single instance of the model
        updated_model = Model(model_name=self.args.model_name,
                              torch_dtype=self.torch_dtype)

        # Initially freeze all parameters
        for param in updated_model.model.parameters():
            param.requires_grad = False

        frozen_model = copy.deepcopy(updated_model)

        # Unfreeze only the layers specified in update_layer_ids for finetuning
        for layer_id in self.args.update_layer_ids:
            if 0 <= layer_id < updated_model.n_layers():
                for param in updated_model.get_layer(layer_id).parameters():
                    param.requires_grad = True

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
        updated_model_activations = self.updated_model.forward(input_ids=x_forget, 
                                                               layer_id=self.args.forget_layer_id,
                                                               no_grad=False).to(self.device, 
                                                                                 dtype=self.torch_dtype)

        # Ensure the control_vector is broadcastable to the shape of updated_model_activations
        control_vector = control_vector.expand_as(updated_model_activations)

        # By default the torch mse_loss function divides the sum of the output by the number of elements in it
        return torch.nn.functional.mse_loss(updated_model_activations, control_vector * self.args.steering_coefficient)
    

    def retain_loss(self, 
                    x_retain):
        """Calculates the retain loss. 
        Args:
            x_retain (torch.Tensor): The input tokens.
        """
        updated_model_activations = self.updated_model.forward(input_ids=x_retain, 
                                                               layer_id=self.args.forget_layer_id,
                                                               no_grad=False).to(self.device, 
                                                                                 dtype=self.torch_dtype)
        frozen_model_activations = self.frozen_model.forward(input_ids=x_retain,
                                                             layer_id=self.args.forget_layer_id, 
                                                             no_grad=True).to(self.device, 
                                                                              dtype=self.torch_dtype)

        # By default the torch mse_loss function divides the sum of the output by the number of elements in it
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
                    print([n.dataset_name for n in self.forget_datasets])
                    
                    x_forget = self.forget_datasets[dataset_id][element_id]['input_ids']
                    x_retain = self.retain_datasets[dataset_id][element_id]['input_ids']
                    control_vector = self.control_vector_list[dataset_id].to(self.device, dtype=self.torch_dtype)
        
                    l_forget = self.forget_loss(x_forget=x_forget, control_vector=control_vector)
                    l_retain = self.args.alpha * self.retain_loss(x_retain=x_retain)
                    full_loss = l_forget + l_retain

                    self.optimizer.zero_grad()
                    full_loss.backward()
                    self.optimizer.step()
                    print(f"Step {batch_id}: full_loss={full_loss.item():.5g}, forget_loss={l_forget.item():.5g}, retain_loss={l_retain.item():.5g}")

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