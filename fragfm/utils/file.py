from pprint import pprint

import pandas as pd
import yaml
from easydict import EasyDict as edict


def read_guacamol_smiles_fn_to_smiles_list(file_path: str) -> list:
    lines = []
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if "get" in line:
                line = line.replace("get", "")
            lines.append(line)  # Remove newline characters
    return lines


def read_coconut_fn_to_smiles_list(file_path: str) -> list:
    lines = []
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if "get" in line:
                line = line.replace("get", "")
            lines.append(line)  # Remove newline characters
    return lines


def read_moses_fn_to_smiles_list(file_path):
    df = pd.read_csv(file_path)
    return df["SMILES"].tolist()


def read_yaml_as_easydict(file_path):
    with open(file_path) as f:
        yaml_data = yaml.safe_load(f)  # Load YAML data
    return edict(yaml_data)  # Convert to EasyDict


def write_easydict_as_yaml(easydict_obj, file_path):
    # Convert the EasyDict object to a regular dict
    data_dict = dict(easydict_obj)

    # Save the dictionary as YAML
    with open(file_path, "w") as f:
        yaml.safe_dump(data_dict, f, default_flow_style=False)


def add_prefix_to_dict_key(dictionary, prefix=""):
    return {f"{prefix}{key}": value for key, value in dictionary.items()}
