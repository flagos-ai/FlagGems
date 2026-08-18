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
import triton

from flag_gems.ops.index_reduce import (
    _index_is_unique,
    _index_reduce_scan_kernel,
    _index_reduce_unique_kernel,
    _reduce_id,
    _restore_dim,
    _validate_args,
)
from flag_gems.ops.index_reduce import index_reduce_ as _generic_index_reduce
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress

logger = logging.getLogger(__name__)


def _index_reduce_scan(inp, dim, index, source, reduce_id, include_self):
    inp_work = dim_compress(inp, dim)
    source_work = dim_compress(source, dim)
    n = index.numel()
    out_n = inp_work.size(-1)
    compute_dtype = torch.float64 if inp_work.dtype == torch.float64 else torch.float32
    inp_compute = inp_work.to(compute_dtype)
    source_compute = source_work.to(compute_dtype)
    out = torch.empty_like(inp_compute)
    total = inp_compute.numel()
    with torch_device_fn.device(inp.device):
        _index_reduce_scan_kernel[(total,)](
            out,
            index,
            source_compute,
            inp_compute,
            total,
            n,
            out_n,
            reduce_id,
            include_self,
            compute_dtype == torch.float64,
        )
    return _restore_dim(out.to(inp.dtype), inp, dim)


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_METAX INDEX_REDUCE_")

    if reduce == "mean" and inp.dtype == torch.bfloat16:
        _validate_args(inp, dim, index, source, reduce)
        if index.numel() == 0:
            return inp
        inp_compute = inp.to(torch.float32)
        source_compute = source.to(torch.float32)
        _generic_index_reduce(
            inp_compute,
            dim,
            index,
            source_compute,
            reduce,
            include_self=include_self,
        )
        inp.copy_(inp_compute.to(inp.dtype))
        return inp

    if reduce != "prod":
        return _generic_index_reduce(
            inp, dim, index, source, reduce, include_self=include_self
        )

    _validate_args(inp, dim, index, source, reduce)
    if index.numel() == 0:
        return inp

    dim = dim % inp.ndim
    index = index.contiguous()
    reduce_id = _reduce_id(reduce)

    if not _index_is_unique(index, inp.size(dim)):
        return _index_reduce_scan(inp, dim, index, source, reduce_id, include_self)

    inp_work = dim_compress(inp, dim)
    source_work = dim_compress(source, dim)
    n = index.numel()
    out_n = inp_work.size(-1)
    m = source_work.numel() // n
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(inp.device):
        _index_reduce_unique_kernel[grid](
            inp_work,
            index,
            source_work,
            m,
            n,
            out_n,
            reduce_id,
            include_self,
        )
    return _restore_dim(inp_work, inp, dim)
