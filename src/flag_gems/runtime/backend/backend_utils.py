import functools
import os
from dataclasses import dataclass

import yaml


# Metadata template,  Each vendor needs to specialize instances of this template
@dataclass
class VendorInfoBase:
    vendor_name: str
    device_name: str
    device_query_cmd: str
    dispatch_key: str = None
    triton_extra_name: str = None


def get_tune_config(vendor_name=None, file_mode="r", file_path=None):
    BACKEND_EVENT = file_path is not None
    config = None
    try:
        if not file_path:
            vendor_name = "_" + vendor_name
            script_path = os.path.abspath(__file__)
            base_dir = os.path.dirname(script_path)
            file_path = os.path.join(base_dir, vendor_name, "tune_configs.yaml")
        else:
            file_path = os.path.join(file_path, "tune_configs.yaml")
        with open(file_path, file_mode) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        if not BACKEND_EVENT:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    return config


@functools.lru_cache(maxsize=None)
def _load_expand_config(file_path, file_mode="r"):
    with open(file_path, file_mode) as file:
        return yaml.safe_load(file) or {}


def get_expand_config(op_name=None, file_mode="r", file_path=None):
    if not file_path:
        raise ValueError("expand config file path is required")
    try:
        config = _load_expand_config(file_path, file_mode)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    if op_name is None:
        return config
    return config.get(op_name)
