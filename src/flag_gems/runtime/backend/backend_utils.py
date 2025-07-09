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


def get_tune_config(vendor_name, file_mode="r"):
    try:
        vendor_name = "_" + vendor_name
        script_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(script_path)
        file_path = os.path.join(base_dir, vendor_name, "tune_configs.yaml")
        with open(file_path, file_mode) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    return config
