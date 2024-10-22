import json
import torch
from torch.utils.data import Dataset
import os
from transformers import AutoTokenizer


class JSONLDataset(Dataset):

    def __init__(self,
                 dataset_name: str,
                 tokenizer: AutoTokenizer,
                 tokenizer_max_length: int = 768,
                 dataset_folder: str = 'data/',
                 batch_size: int = 4,
                 min_len: int = 50
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.min_len = min_len
        self.data = self.load_jsonl()


    def load_jsonl(self):
        file_path = os.path.join(self.dataset_folder, self.dataset_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        data = []
        for line in open(f"{file_path}", "r"):
            if "bio-forget-corpus" in self.dataset_name:
                raw_text = json.loads(line)['text']
            else:
                raw_text = line
            if len(raw_text) > self.min_len:
                data.append(str(raw_text))
        data = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
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
        try:
            if isinstance(item, str):
                # If the item is a string, parse it as JSON
                item = json.loads(item)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at index {idx}")
            print(f"Raw item: {item}")
            print(f"Error details: {str(e)}")
            raise 
        
        inputs = self.tokenizer(item, 
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.tokenizer_max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0), # Squeeze to remove batch dim
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }