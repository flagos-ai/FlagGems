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

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from backend_utils import VendorDescriptor  # noqa: E402

# Detect the tsingmicro chip generation from the installed device SDK:
#   torch_txda  -> arch "1" -> TX81
#   torch_raisa -> arch "2" -> TX82
# The per-chip operators/configs live in the fully isolated tx81/ and tx82/
# directories and are selected through ARCH_MAP by the runtime arch event.
try:
    import torch_txda  # noqa: F401

    arch = "1"
except ImportError:
    import torch_raisa  # noqa: F401

    arch = "2"

os.environ["ARCH"] = str(arch)
ARCH_MAP = {"1": "tx81", "2": "tx82"}

# Torch device attribute and chip name of the detected generation.
_ARCH_TORCH_DEVICE = {"1": "txda", "2": "raisa"}
_ARCH_CHIP_NAME = {"1": "TX81", "2": "TX82"}

SPM_SIZE = 3 * 1024 * 1024
SYS_SPM_RESERVED_SIZE = 64 * 1024
OPS_SPM_RESERVED_SIZE = 256


@dataclass
class TxdaDeviceProperties:
    name: str
    major: int
    minor: int
    total_memory: int  # MB
    multi_processor_count: int
    uuid: str
    L2_cache_size: int  # MB

    def __repr__(self) -> str:
        return (
            f"TxdaDeviceProperties(name='{self.name}', major={self.major}, "
            f"minor={self.minor}, total_memory={self.total_memory}MB, "
            f"multi_processor_count={self.multi_processor_count}, "
            f"uuid={self.uuid}, L2_cache_size={self.L2_cache_size}MB)"
        )


def get_device_properties(
    device: torch.device | str | int | None = None,
) -> TxdaDeviceProperties:
    return TxdaDeviceProperties(
        name=_ARCH_CHIP_NAME[arch],
        major=8,
        minor=1,
        total_memory=64 * (1024**3),  # 64GB
        multi_processor_count=16,
        uuid="",
        L2_cache_size=3 * (1024**2),  # 3MB
    )


def get_device_capability(
    device: Optional[Union[torch.device, str, int]] = None,
) -> Tuple[int, int]:
    return (8, 0)


_torch_arch_device = getattr(torch, _ARCH_TORCH_DEVICE[arch], None)
if _torch_arch_device is not None:
    if not hasattr(_torch_arch_device, "get_device_properties"):
        setattr(_torch_arch_device, "get_device_properties", get_device_properties)

    if not hasattr(_torch_arch_device, "get_device_capability"):
        setattr(_torch_arch_device, "get_device_capability", get_device_capability)

vendor_info = VendorDescriptor(
    vendor_name="tsingmicro",
    device_name=_ARCH_TORCH_DEVICE[arch],
    device_query_cmd="tsm_smi",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    int64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "clone",
    "sort",
    "sort_stable",
    "expand",
    "expand_",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "lt_scalar",
    "lt",
    "masked_fill",
    "masked_fill_",
    "le_scalar",
    "le",
    "scatter",
    "scatter_",
)

__all__ = ["*"]
