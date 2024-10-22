import yaml
import argparse
import json
import numpy as np
import torch

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert the dictionary to an argparse.Namespace object
    args = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(args, sub_key, sub_value)
        else:
            setattr(args, key, value)
    
    return args

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
        if callable(obj):
            return obj.__name__
        return super(CustomJSONEncoder, self).default(obj)