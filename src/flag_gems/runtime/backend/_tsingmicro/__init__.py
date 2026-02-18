from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch_txda  # noqa: F401
from backend_utils import VendorInfoBase  # noqa: E402


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
        name="TX81",
        major=8,
        minor=1,
        total_memory=64 * 1024 ^ 3,  # 64GB
        multi_processor_count=16,
        uuid="",
        L2_cache_size=3 * 1024 ^ 2,  # 3MB
    )


def get_device_capability(
    device: Optional[Union[torch.device, str, int]] = None
) -> Tuple[int, int]:
    return (8, 0)


if not hasattr(torch.txda, "get_device_properties"):
    setattr(torch.txda, "get_device_properties", get_device_properties)

if not hasattr(torch.txda, "get_device_capability"):
    setattr(torch.txda, "get_device_capability", get_device_capability)

vendor_info = VendorInfoBase(
    vendor_name="tsingmicro",
    device_name="txda",
    device_query_cmd="tsm_smi",
    dispatch_key="PrivateUse1",
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
