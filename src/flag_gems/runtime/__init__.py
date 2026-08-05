# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

from . import backend, common, error
from .backend.device_finder import DeviceDetector
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
