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

import importlib.util
from typing import Any

from backend_utils import VendorDescriptor  # noqa: E402

if importlib.util.find_spec("triton.backends.spine_triton") is not None:
    from .utils.config_pre_hook import setup_triton_config

    setup_triton_config()

    import triton  # noqa: E402
    from triton.backends.spine_triton.driver import CPUDriver  # noqa: E402

    triton.runtime.driver.set_active(CPUDriver())  # noqa: E402


vendor_info = VendorDescriptor(
    vendor_name="spacemit",
    device_name="cpu",
    device_query_cmd="spacemit-tcm-smi",
    fp64_enabled=False,
    bf16_enabled=False,
    int64_enabled=False,
)


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = self.idx

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = self.prev_idx
        return False


class _DeviceWrapper:
    def __init__(self, device: Any): ...

    def __enter__(self): ...

    def __exit__(self, type: Any, value: Any, traceback: Any):
        ...
        return False

    @staticmethod
    def current_device():
        """Return device index for kernel cache. CPU backend always uses device 0."""
        return 0


CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]
