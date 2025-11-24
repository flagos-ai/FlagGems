from . import backend, commom_utils, error
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
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


def replace_customized_ops(_globals):
    if device.vendor != commom_utils.vendors.NVIDIA:
        customized_op_infos = backend.get_current_device_extend_op(device.vendor_name)
        try:
            for fn_name, fn in customized_op_infos:
                _globals[fn_name] = fn
        except RuntimeError as e:
            error.customized_op_replace_error(e)


__all__ = ["*"]
