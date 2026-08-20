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

from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)


def _compute_flat_offset(shape, strides, dim, N):
    idx = torch.arange(N, device="cpu", dtype=torch.int64)
    coord = torch.empty((len(shape), N), dtype=torch.int64, device="cpu")
    for i in reversed(range(len(shape))):
        coord[i] = idx % shape[i]
        idx = idx // shape[i]
    offset = torch.zeros(N, dtype=torch.int64, device="cpu")
    for i in range(len(shape)):
        if i != dim:
            offset += coord[i] * strides[i]
    return offset


def scatter_add_(inp, dim, index, src):
    logger.debug("GEMS_ASCEND SCATTER_ADD_")
    out = inp
    dim = dim % inp.ndim
    dim_stride = inp.stride(dim)

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)

    N = index.numel()
    if N == 0:
        return out

    flat_index = index.reshape(-1).to(torch.int64).cpu()
    flat_src = src_strided.reshape(-1).contiguous().cpu()
    out_cpu = out.cpu()

    base_offset = _compute_flat_offset(index.shape, inp_restrided.stride(), dim, N).to(
        torch.int64
    )

    flat_out = out_cpu.reshape(-1)
    for i in range(N):
        idx = flat_index[i].item()
        out_offset = base_offset[i].item() + idx * dim_stride
        flat_out[out_offset] += flat_src[i].item()

    inp.copy_(out_cpu.to(inp.device))
    return inp
