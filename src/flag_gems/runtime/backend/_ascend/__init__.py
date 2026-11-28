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

import os

from backend_utils import VendorDescriptor  # noqa: F402

from .utils import CORE_NUM  # noqa: F401

# Arch key -> directory under ``_ascend`` holding that arch's specialized
# operators.
#
#   Atlas A5 (Ascend950DT / Ascend950PR, NpuArch DAV_3510) -> a5/
#
# Atlas A2/A3 (Ascend910B*, Ascend910_93*) have no directory of their own: they
# are the baseline the generic ``_ascend/ops`` kernels target.
ARCH_MAP = {"a5": "a5"}

# Markers matched as a substring of the SoC name reported by the driver. That
# name carries a SKU suffix (``Ascend950DT_9582``, ``Ascend950PR``), so one
# entry covers every SKU of a generation.
_SOC_ARCH_MARKERS = (("950", "a5"),)

# Accepted ``ARCH`` spellings per arch key, so a machine without the device can
# still exercise an arch.
_ARCH_ALIASES = {
    "a5": {"a5", "ascend_a5", "950", "ascend_950"},
}


def get_arch_key(torch_device_object, device=0):
    """``ARCH_MAP`` key for the current NPU, or None when it has no arch dir.

    ``BackendArchEvent.get_arch`` calls this instead of its generic
    ARCH/device-property lookup: the Ascend generations cannot be told apart
    that way (``props.major`` is None on an NPU, and the leading digit of the
    SoC name is ``9`` for both Ascend910B and Ascend950). ``ARCH`` wins when
    set; otherwise the SoC name decides. Returns None when no device is
    reachable (CPU-only build, driver not up), which keeps the generic kernels
    in place.
    """
    arch = os.environ.get("ARCH", "").strip().lower()
    if arch:
        for arch_key, aliases in _ARCH_ALIASES.items():
            if arch in aliases:
                return arch_key
    soc_name = arch or _get_device_name(torch_device_object, device)
    if not soc_name:
        return None
    for marker, arch_key in _SOC_ARCH_MARKERS:
        if marker in soc_name:
            return arch_key
    return None


def _get_device_name(torch_device_object, device=0):
    """SoC name reported by the driver, e.g. ``Ascend950DT_9582``."""
    try:
        if torch_device_object is None:
            # The arch event normally runs after the runtime has built the
            # ``torch.npu`` handle; reach it directly if it has not yet.
            import torch
            import torch_npu  # noqa: F401

            torch_device_object = torch.npu
        if not torch_device_object.is_available():
            return ""
        return (torch_device_object.get_device_name(device) or "").lower()
    except Exception:
        return ""


def get_triton_extra_name():
    try:
        import triton
        from packaging import version

        if version.parse(triton.__version__) < version.parse("3.2.0"):
            return "ascend"
        else:
            return "cann"
    except Exception:
        return "ascend"


# FP8/FNUZ capability probes belong to selected scaled_mm tests, not backend import.
vendor_info = VendorDescriptor(
    vendor_name="ascend",
    device_name="npu",
    device_query_cmd="npu-smi info",
    dispatch_key="PrivateUse1",
    triton_extra_name=get_triton_extra_name(),
    fp64_enabled=False,
    complex_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = (
    "to_copy",
    "contiguous",
    "copy_",
    "_to_copy",
    "sort",
    "sort_stable",
    "topk",
    "exponential_",  # TODO: Err occurred when running Qwen3.6 vLLM with flagtree ascend3.5
)


__all__ = ["*"]
