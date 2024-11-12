import json
import torch
from torch.utils.data import Dataset
import os
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from datasets import load_dataset
from typing import List, Dict, Optional, Iterator, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):

    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 tokenizer_max_length: int = 768,
                 batch_size: int = 4,
                 min_len: int = 50
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.batch_size = batch_size
        self.min_len = min_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self._load_data()


    @abstractmethod
    def _load_data(self):
        pass


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        inputs = self.tokenizer(item, 
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.tokenizer_max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


class JSONLDataset(BaseDataset):
    """
    Dataset class for loading and processing JSONL format data.

    Attributes:
        dataset_name (str): Name of the dataset from the data folder
        dataset_folder (str): Name of the folder where the dataset is located
    """

    def __init__(self, 
                 dataset_name: str, 
                 dataset_folder: str = 'data/',
                 **kwargs
    ) -> None:
        """Initialize the JSONL dataset."""
        self.dataset_name = dataset_name
        self.dataset_folder = dataset_folder
        super().__init__(**kwargs)


    def _load_data(self) -> List[Dict]:
        """
        Load and parse the JSONL file.

        Returns:
            List of dictionaries containing parsed data entries.
        """
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
        return [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]


class WikitextDataset(BaseDataset):
    """
    Dataset class for handling the Wikitext dataset from HuggingFace.

    Attributes:
        dataset_version (str): Name of the version of wikitext to use
    """

    def __init__(self, 
                 dataset_version: str = 'wikitext-2-raw-v1', 
                 **kwargs
    ) -> None:
        self.dataset_version = dataset_version
        super().__init__(**kwargs)


    def _load_data(self) -> List[str]:
        """Load data from Wikitext dataset."""

        dataset = load_dataset("wikitext", self.dataset_version, split="test")
        if dataset is None:
            raise DatasetError("Failed to load Wikitext dataset")
        data = [item['text'] for item in dataset if len(item['text']) > self.min_len]
        return [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]