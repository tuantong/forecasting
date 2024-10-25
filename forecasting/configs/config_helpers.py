from pathlib import Path
from typing import Dict, List, Union

import yaml


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""


def parse_yaml_as_config(yaml_path: Union[str, Path], yaml_str: str = ""):
    if yaml_path and yaml_str:
        raise OSError("Only one of (yaml_path, yam_str) is allowed to be not None.")

    if yaml_path:
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:  # yaml_str is not None
        config = yaml.safe_load(yaml_str)

    return config


def save_yaml_config_to_file(
    yaml_config: Dict,
    yaml_path: Union[str, Path],
):
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(yaml_config, yaml_file, default_flow_style=False)


def validate_config(config_data: Dict, required_fields: List):
    for field in required_fields:
        if field not in config_data:
            raise ConfigValidationError(f"Missing required field: {field}")
