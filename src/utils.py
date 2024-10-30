import os
import sys

import re
import json
import yaml

from datetime import datetime

import random
import numpy as np
import torch


def extract_folders_from_dir(directory):
    """
    Generator function that yields the names of folders in the given directory.

    Args:
    - directory (str): The path to the directory.

    Yields:
    - str: The name of a folder.
    """
    for entry in os.scandir(directory):
        if entry.is_dir():
            yield entry.name
# Example usage:
# for folder_name in extract_folders_from_dir('path/to/directory'):
#     print(folder_name)


def extract_files_from_dir(directory:str, sort=False):
    """
    Generator function that yields the content of each text document in the given directory.

    Args:
    - directory (str): The path to the directory containing text documents.

    Yields:
    - str: The content of a text document.
    """
    list_dir = os.listdir(directory)
    if sort: list_dir = sorted(list_dir)
    for filename in list_dir:
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                yield filename, file.read()
# Example usage:
# for content in extract_documents_from_dir('path/to/text/documents'):
#     print(content)


# def find_single_number_in_string(s):
#     # Use regular expression to find numbers
#     numbers = re.findall(r'\d+', s)
#     # Check if there are multiple numbers
#     if len(numbers) != 1:
#         raise ValueError("Multiple numbers found" if len(numbers) > 1 else "No numbers found")
#     return numbers[0]

def find_single_number_in_string(s):
    matches = re.findall(r'(\S+) - ', s)
    if matches:
        id = matches[0]  # Take only the first match
        return id
    else:
        raise Exception(f'Could not find an id number in string "{s}".')


# def load_json_file(file_path):
#     with open(file_path, 'r') as file:
#         # Parse JSON file into a Python dictionary
#         data = json.load(file)
#     return data
def load_json_file(file_path:str, accept_none=False) -> dict:
    if not file_path.endswith(".json"): file_path += ".json"
    if not os.path.isfile(file_path):
        if accept_none:
            return None
        else:
            raise FileNotFoundError(file_path)
    try:
        with open(file_path, 'r') as json_file:
            data_dict = json.load(json_file)
    except json.JSONDecodeError:
        if accept_none:
            data_dict = None
        else:
            raise Exception(f'Could not open file "{json_file}"')
    return data_dict


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists


def extract_unique_keys(dict_list):
    """
    Extract keys that have different values or do not exist in all dictionaries within the list.

    :param dict_list: List of dictionaries to analyze
    :return: Set of unique keys
    """
    if not dict_list:  # If the list is empty, return an empty set
        return set()

    # Extract the keys of the first dictionary to start comparison
    unique_keys = set(dict_list[0].keys())
    common_keys = set(unique_keys)

    # Compare with the rest of the dictionaries
    for d in dict_list[1:]:
        current_keys = set(d.keys())
        # Update unique keys with keys that are not in the current set of common keys
        unique_keys.update(current_keys.symmetric_difference(common_keys))
        # Update common keys with keys that are common in all dictionaries so far
        common_keys.intersection_update(current_keys)

    # Remove keys that are common to all dictionaries
    unique_keys.difference_update(common_keys)

    # Check for keys that are the same across dictionaries but have different values
    for key in common_keys:
        value_set = set(d.get(key, None) for d in dict_list)
        if len(value_set) > 1:
            unique_keys.add(key)

    return unique_keys


def load_text_file(file_path):
    """
    Load the content of a text file and return it as a string.

    Args:
    file_path (str): The path to the text file.

    Returns:
    str: The content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
  

def evaluate_keywords():
  pass


def has_duplicates(lst):
    seen = set()
    for item in lst:
        if item in seen:
            return True, item
        seen.add(item)
    return False, None


def save_dict_as_json(data_dict, file_path, create_folders=False, indent=4):
    """
    Saves a dictionary as a JSON file at the specified file path.
    Creates intermediate directories if they do not exist.

    Args:
    data_dict (dict): The dictionary to be saved.
    file_path (str): The path where the JSON file will be saved.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if (not os.path.exists(directory)) and create_folders:
        os.makedirs(directory)
    try:
        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=indent)
    except Exception as e:
        print(f"Error saving file: {e}")


def set_all_seeds(seed:int, device='cpu'):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if device=='cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # torch.backends.cudnn.enabled       = False
    elif device=='cpu':
        pass
    else:
        raise ValueError(f'Device {device} not supported.')


def set_reproducible_experiment(seed, detect_anomaly=False, device='cpu'):
    torch.use_deterministic_algorithms(mode=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    set_all_seeds(seed=seed, device=device)


def current_datetime():
    return datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%S")



def to_sweep_format(parameters:dict):
    sweep_parameters = {}
    for key,param in parameters.items():
        if isinstance(param, dict):
            raise Exception(f'Dict param not supported')
        elif isinstance(param,list):
            assert len(param) > 0
            if len(param) == 1:
                sweep_parameters[key] = {"value":param[0]}
            else: 
                sweep_parameters[key] = {"values":param}
        else:
            sweep_parameters[key] = {"value":param}
    return sweep_parameters



def retrieve_arguments():
    arguments = []
    for arg in sys.argv[1:]:
        arguments.append(arg)
    return arguments


def load_yaml_file(filepath:str):
    if not '.yaml' in filepath:
        filepath = filepath + '.yaml'
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data