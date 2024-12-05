from . import backend, commom_utils
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
device = DeviceDetector()


def get_triton_config(op_name):
    return config_loader.get_triton_config(op_name)


def get_device_fn(api_name):
    return backend.gen_torch_device_fn(api_name)


__all__ = ["commom_utils", "backend", "device", "get_triton_config"]
