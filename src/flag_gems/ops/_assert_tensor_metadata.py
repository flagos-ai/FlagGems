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
        raise RuntimeError(f"Tensor dtype mismatch! Expected: {dtype}, Got: {a.dtype}")
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
