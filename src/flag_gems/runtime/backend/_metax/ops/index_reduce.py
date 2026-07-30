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

import triton

from flag_gems.ops.index_reduce import (
    _index_is_unique,
    _index_reduce_unique_kernel,
    _reduce_id,
    _restore_dim,
    _validate_args,
)
from flag_gems.ops.index_reduce import index_reduce_ as _generic_index_reduce
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress

logger = logging.getLogger(__name__)


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_METAX INDEX_REDUCE_")

    if reduce != "prod":
        return _generic_index_reduce(
            inp, dim, index, source, reduce, include_self=include_self
        )

    _validate_args(inp, dim, index, source, reduce)
    if index.numel() == 0:
        return inp

    dim = dim % inp.ndim
    index = index.contiguous()
    if not _index_is_unique(index, inp.size(dim)):
        return _generic_index_reduce(
            inp, dim, index, source, reduce, include_self=include_self
        )

    inp_work = dim_compress(inp, dim)
    source_work = dim_compress(source, dim)
    n = index.numel()
    m = source_work.numel() // n
    out_n = inp_work.size(-1)
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
            _reduce_id(reduce),
            include_self,
        )
    return _restore_dim(inp_work, inp, dim)
