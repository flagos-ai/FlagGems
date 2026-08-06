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


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    logger.debug("GEMS PACK_SEQ_TRITON")
    if x.dtype == torch.uint8:
        assert (
            isinstance(pad_value, int) and 0 <= pad_value <= 255
        ), f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
    else:
        float(pad_value)

    lengths_list = lengths.cpu().tolist()
    original_shape = x.shape
    x_flat = x.reshape(original_shape[0], -1).cpu()
    out = torch.full(
        (len(lengths_list), max(lengths_list), x_flat.shape[1]),
        pad_value,
        dtype=torch.float32,
    )
    offset = 0
    for batch, seq_len in enumerate(lengths_list):
        out[batch, :seq_len] = x_flat[offset : offset + seq_len]
        offset += seq_len

    out = out.to(device=x.device).to(dtype=x.dtype)
    if len(original_shape) > 2:
        output_shape = (len(lengths_list), max(lengths_list)) + original_shape[1:]
        return out.reshape(output_shape)
    return out
