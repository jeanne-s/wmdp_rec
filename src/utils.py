import yaml
import argparse

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