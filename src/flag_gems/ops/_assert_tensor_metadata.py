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

import logging

import torch

logger = logging.getLogger(__name__)

# This op validates tensor metadata on the host: it compares the input
# tensor's size, stride, dtype, device and layout against expected values
# and raises RuntimeError on any mismatch. It performs no device-side
# computation, so there is no Triton kernel.

# PyTorch ScalarType display names, matching the native aten error wording
# used on the "Expected" side (e.g. "Expected: Half").
_DTYPE_NAMES = {
    torch.float16: "Half",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.bfloat16: "BFloat16",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.uint8: "Byte",
    torch.bool: "Bool",
    torch.complex64: "ComplexHalf",
    torch.complex128: "ComplexFloat",
}


def _assert_tensor_metadata(
    a: torch.Tensor,
    size=None,
    stride=None,
    dtype=None,
    *,
    device=None,
    layout=None,
):
    logger.debug("GEMS _ASSERT_TENSOR_METADATA")
    if size is not None:
        if a.size() != torch.Size(size):
            raise RuntimeError(
                f"Tensor sizes mismatch! Expected: {list(size)}, Got: {list(a.size())}"
            )
    if stride is not None:
        if a.stride() != tuple(stride):
            raise RuntimeError(
                f"Tensor strides mismatch! Expected: {list(stride)}, Got: {list(a.stride())}"
            )
    if dtype is not None and a.dtype != dtype:
        expected = _DTYPE_NAMES.get(dtype, str(dtype))
        got = _DTYPE_NAMES.get(a.dtype, str(a.dtype))
        raise RuntimeError(f"Tensor dtype mismatch! Expected: {expected}, Got: {got}")
    if device is not None:
        expected_device = torch.device(device)
        actual_device = a.device
        if expected_device.type != actual_device.type or (
            expected_device.index is not None
            and expected_device.index != actual_device.index
        ):
            raise RuntimeError(
                f"Tensor device mismatch! Expected: {expected_device}, "
                f"Got: {actual_device}"
            )
    if layout is not None and a.layout != layout:
        raise RuntimeError(
            f"Tensor layout mismatch! Expected: {layout}, Got: {a.layout}"
        )
    return None
