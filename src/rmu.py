import numpy as np
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM
import tqdm as tqdm

 

class BaseRMU: 
    
    def __init__(self, args):
        self.args = args
        #updated_model, frozen_model, retain_dataset, forget_dataset_list: list[str], c, alpha
    

    def setup(self):
        """
        This method sets up the tokenizer and optimizer for the model.
        """
        self.tokenizer = self.load_tokenizer()
        self.updated_model, self.frozen_model = self.load_models()
        self.optimizer = self.create_optimizer() 
        self.control_vector_list = self.create_control_vector_list()
        

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, 
                                                  trust_remote_code_true=True, 
                                                  use_fast=False)
        return tokenizer 


    def load_models(self):
        updated_model = AutoModelForCausalLM.from_pretrained(self.args.model_name)
        frozen_model = AutoModelForCausalLM.from_pretrained(self.args.model_name)
        return updated_model, frozen_model


    def create_optimizer(self):
        optimizer = AdamW()#TODO
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


    def finetune():
        """Main training loop."""
        
        for dataset_id, forget_dataset in enumerate(self.args.forget_dataset_list):
            for epoch in range(self.args.num_epochs):
                for batch_id in range(self.args.num_batches):
                
                    x_forget = #get_sample()
                    x_retain = 
            
                    l_forget = self.forget_loss(x_forget=x_forget, control_vector=self.control_vector_list[dataset_id])
                    l_retain = self.retain_loss(x_retain=x_retain)
                    full_loss = l_forget + self.args.alpha * l_retain

