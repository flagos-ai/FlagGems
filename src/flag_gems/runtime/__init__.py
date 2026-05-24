from contextlib import contextmanager

from . import backend, common, error
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
if device.name == "cpu":
    if not hasattr(torch_device_fn, "device"):

        @contextmanager
        def _noop_device_guard(_device=None):
            yield

        torch_device_fn.device = _noop_device_guard
    if not hasattr(torch_device_fn, "_DeviceGuard"):

        class _NoOpDeviceGuard:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        torch_device_fn._DeviceGuard = _NoOpDeviceGuard

# torch_backend_device is like 'torch.backend.cuda' object
torch_backend_device = backend.get_torch_backend_device_fn()


def get_tuned_config(op_name):
    return config_loader.get_tuned_config(op_name)


def get_heuristic_config(op_name):
    return config_loader.get_heuristics_config(op_name)


def replace_customized_ops(_globals):
    event = backend.BackendArchEvent()
    arch_specific_ops = event.get_arch_ops() if event.has_arch else None
    extended_ops = backend.get_customized_ops(device.vendor_name)
    if device.vendor != common.vendors.NVIDIA:
        try:
            for fn_name, fn in extended_ops:
                _globals[fn_name] = fn
        except RuntimeError as e:
            error.customized_op_replace_error(e)
    if arch_specific_ops:
        try:
            for fn_name, fn in arch_specific_ops:
                _globals[fn_name] = fn
        except RuntimeError as e:
            error.customized_op_replace_error(e)


def get_expand_config(op_name, yaml_path=None):
    return config_loader.get_expand_config(op_name=op_name, yaml_path=yaml_path)


def ops_get_configs(op_name, pre_hook=None, yaml_path=None):
    return config_loader.ops_get_configs(
        op_name=op_name,
        pre_hook=pre_hook,
        yaml_path=yaml_path,
    )


__all__ = ["*"]
