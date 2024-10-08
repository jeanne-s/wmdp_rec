# TODO: we should check that there is no truncation problem

import json
import torch
from torch.utils.data import Dataset
import os
from transformers import AutoTokenizer


class JSONLDataset(Dataset):

    def __init__(self,
                 dataset_name: str,
                 tokenizer: AutoTokenizer,
                 dataset_folder: str = '../datasets/'): # TODO: that argument could be better handled
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.dataset_folder = dataset_folder
        self.data = self.load_jsonl()


    def load_jsonl(self):
        """Loads data from the .jsonl file into memory."""
        file_path = os.path.join(self.dataset_folder, self.dataset_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        data = []
        with open(file_path, 'r') as f:
            for line in f:
                json_line = json.loads(line.strip())
                data.append(json_line)
        return data


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)


    def __getitem__(self, idx):
        """
        Fetches the item (input and label) at the given index.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: A dictionary containing tokenized input and the label.
        """
        item = self.data[idx]
        
        title = item['title']
        abstract = item['abstract']
        text = item['text']
        divmod = item['doi']
        
        inputs = self.tokenizer(text, 
                                return_tensors="pt",
                                padding=True,
                                truncation=False)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0), # Squeeze to remove batch dim
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
