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

from flag_gems.ops.index_reduce import (
    _index_reduce_scan_kernel,
    _reduce_id,
    _restore_dim,
    _validate_args,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress

logger = logging.getLogger(__name__)


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_ASCEND INDEX_REDUCE_")
    _validate_args(inp, dim, index, source, reduce)

    if index.numel() == 0:
        return inp

    dim = dim % inp.ndim
    index = index.contiguous()
    reduce_id = _reduce_id(reduce)
    inp_work = dim_compress(inp, dim)
    source_work = dim_compress(source, dim)

    N = index.numel()
    out_n = inp_work.size(-1)

    inp_compute = inp_work.to(torch.float32)
    source_compute = source_work.to(torch.float32)
    out = torch.empty_like(inp_compute)
    total = inp_compute.numel()
    with torch_device_fn.device(inp.device):
        _index_reduce_scan_kernel[(total,)](
            out,
            index,
            source_compute,
            inp_compute,
            total,
            N,
            out_n,
            reduce_id,
            include_self,
            False,
        )
    return _restore_dim(out.to(inp.dtype), inp, dim)
