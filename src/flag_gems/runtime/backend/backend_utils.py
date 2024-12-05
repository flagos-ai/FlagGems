import os
from dataclasses import dataclass

import yaml


# Metadata template,  Each vendor needs to specialize instances of this template
@dataclass
class VendorInfoBase:
    vendor_name: str
    device_name: str
    device_query_cmd: str


def get_tune_config(vendor_name, file_mode="r"):
    vendor_name = "_" + vendor_name
    script_path = os.path.abspath(__file__)
    file_path = os.path.dirname(script_path) + "/" + vendor_name + "/tune_configs.yaml"
    with open(file_path, file_mode) as file:
        config = yaml.safe_load(file)

    return config
