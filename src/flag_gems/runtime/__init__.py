from . import backend, commom_utils, moduel_tool
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
device = DeviceDetector()
# torch_device_fn is like 'torch.cuda' object
torch_device_fn = backend.gen_torch_device_object()
tl_extra_module = moduel_tool.tl_extra_module
# torch_backend_device is like 'torch.backend.cuda' object
torch_backend_device = backend.get_torch_backend_device_fn()


def get_triton_config(op_name):
    return config_loader.get_triton_config(op_name)


__all__ = ["commom_utils", "backend", "device", "get_triton_config"]
