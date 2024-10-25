import json
import pickle

import yaml

from forecasting.configs import logger


def save_list_to_txt(lst, local_path):
    """
    Save a list to a file, with each element of the list on a new line.
    Args:
        lst (list): The list of elements to save. Elements will be converted to strings.
        local_path (str): The path to the file where the list should be saved. If the
                          file exists, it will be overwritten; otherwise, a new file
                          will be created.

    Returns:
        None

    Example:
    logger.info(f"Saving list to {local_path}")
    with open(local_path, 'w+', encoding='utf-8') as file:
        file.writelines('\n'.join(map(str, lst)))
    """
    logger.info(f"Saving list to {local_path}")
    with open(local_path, "w+", encoding="utf-8") as file:
        file.writelines("\n".join(map(str, lst)))


def save_dict_to_json(dictionary, local_path):
    """Save a dictionary to a JSON file.

    Args:
        dictionary (dict): The dictionary to save.
        local_path (str): The path to the file where the dictionary should be saved.
    """
    # Assuming `logger` is defined elsewhere and configured for logging
    logger.info(f"Saving dictionary to {local_path}")
    with open(local_path, "w", encoding="utf-8") as file:
        json.dump(dictionary, file, ensure_ascii=False, indent=4)


def load_dict_from_json(local_path):
    """Load and return a dictionary from a JSON file.

    Args:
        local_path (str): The path to the JSON file to be loaded.

    Returns:
        dict: The dictionary loaded from the JSON file.
    """
    with open(local_path, encoding="utf-8") as file:
        return json.load(file)


def load_dict_from_yaml(local_path):
    """
    Load and return a dictionary from a YAML file.

    Args:
        local_path (str): The path to the YAML file to be loaded.

    Returns:
        dict: The dictionary loaded from the YAML file.
    """
    with open(local_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data_from_pickle(local_path):
    with open(local_path, "rb") as file:
        return pickle.load(file)


def save_data_to_pickle(data, local_path):
    with open(local_path, "wb") as file:
        pickle.dump(data, file)
