import numpy as np
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM
import tqdm as tqdm
from dataset import JSONLDataset
from model import Model

 

class BaseRMU: 
    
    def __init__(self, args):
        self.args = args
        #updated_model, frozen_model, retain_dataset, forget_dataset_list: list[str], alpha
    

    def setup(self):
        """
        This method sets up the tokenizer and optimizer for the model.
        """
        self.tokenizer = self.load_tokenizer()
        self.updated_model, self.frozen_model = load_models()
        self.optimizer = self.load_optimizer() 
        self.control_vector_list = self.create_control_vector_list()
        self.retain_datasets, self.forget_datasets = self.setup_datasets()
        

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, 
                                                  trust_remote_code_true=True, 
                                                  use_fast=False)
        return tokenizer 

    
    def load_models(self):
        updated_model = Model(model_name=self.args.model_name)
        frozen_model = Model(model_name=self.args.model_name)
        return updated_model, frozen_model


    def load_optimizer(self): # TODO: Unsure about that function
        optimizer_param_layer_id = set(self.args.optimizer_param_layer_id)
        params = [
            p
            for layer_id in layer_ids
            if 0 <= layer_id < self.updated_model.n_layers()
            for i, p in enumerate(self.updated_model.get_layer(layer_id).parameters())
            if i in optimizer_param_layer_id
        ]
        optimizer = AdamW(params, lr=self.args.learning_rate) 
        return optimizer
    

    def create_control_vector_list(self):
        """ 
        Samples a unit vector with independent entries drawn uniformly at random from [0,1).
        Corresponds to u in the paper. One unit vector is created per forget dataset; 
        for each forget dataset u is held fixed throughout training.
        """
        control_vector_list = []
        for i in range(len(self.args.forget_dataset_list)):
            control_vector = torch.rand(1,1, self.args.updated_model.config.hidden_size,
                                        dtype=self.args.updated_model.dtype,
                                        device=self.args.updated_model.device)
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

        inputs = self.tokenizer(x_forget, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=self.args.max_length) 
        activations = self.forward()
        
        return torch.nn.functional.mse_loss(activations, control_vector)
    

    def retain_loss(self, 
                    x_retain):
        inputs = self.tokenizer(x_retain,
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True,
                                max_length=self.args.max_length)
        
        updated_model_activations = self.forward(self.updated_model, inputs) #other_args
        frozen_model_activations = self.forward(self.frozen_model, inputs) #other_args

        return torch.nn.functional.mse_loss(updated_model_activations, frozen_model_activations)


    def forward():
        """Forward pass"""
        pass


    def finetune(self):
        """Main training loop."""
        
        for epoch in range(self.args.num_epochs):
            with tqdm.tqdm(total=self.args.num_batches) as pbar:
                for batch_id in range(self.args.num_batches):
                
                    dataset_id = batch_id % len(self.forget_datasets) # TODO: would be great if we could proof-read that
                    element_id = batch_id // len(self.forget_datasets)
                    
                    x_forget = self.forget_datasets[dataset_id][element_id]['input_ids']
                    x_retain = self.retain_datasets[dataset_id][element_id]['input_ids']
                    control_vector = self.control_vector_list[dataset_id]
            
                    l_forget = self.forget_loss(x_forget=x_forget, control_vector=control_vector)
                    l_retain = self.retain_loss(x_retain=x_retain)
                    full_loss = l_forget + self.args.alpha * l_retain

                    # gradient descent ...

