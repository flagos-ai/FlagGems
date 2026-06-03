from . import backend, common, error
from .backend.device import DeviceDetector
from .configs_loader import TunedConfigLoader
from .flagtune import flagtune, flagtune_enabled

config_loader = TunedConfigLoader()
device = DeviceDetector()

"""
The dependency order of the sub-directory is strict, and changing the order arbitrarily may cause errors.
"""

# torch_device_fn is like 'torch.cuda' object
backend.set_torch_backend_device_fn(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()

# torch_backend_device is like 'torch.backend.cuda' object
torch_backend_device = backend.get_torch_backend_device_fn()


def get_tuned_config(op_name):
    return config_loader.get_tuned_config(op_name)


def get_heuristic_config(op_name):
    return config_loader.get_heuristics_config(op_name)


def get_expand_config(op_name, yaml_path=None):
    return config_loader.get_expand_config(op_name=op_name, yaml_path=yaml_path)


def ops_get_configs(op_name, pre_hook=None, yaml_path=None):
    return config_loader.ops_get_configs(
        op_name=op_name,
        pre_hook=pre_hook,
        yaml_path=yaml_path,
    )


__all__ = [
    "TunedConfigLoader",
    "DeviceDetector",
    "backend",
    "common",
    "config_loader",
    "device",
    "error",
    "flagtune",
    "flagtune_enabled",
    "get_expand_config",
    "get_heuristic_config",
    "get_tuned_config",
    "ops_get_configs",
    "replace_customized_ops",
    "torch_backend_device",
    "torch_device_fn",
]
