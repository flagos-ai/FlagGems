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
import triton.language as tl

from flag_gems.ops.copy import copy_ as _gems_copy_

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic
from .copy import _copy_flat_kernel, _pick_flat_block
from .expand_copy import _launch_bcast

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

_INT32_MAX = 2**31 - 1
_B8 = 65536  # int64 lanes per program (8 bytes each => 512 KiB/program)
_BE = 1024  # elementwise fallback block
_MAX_FLAT_DIM = 6


@triton.jit
def _copy_i64_exact(x_ptr, y_ptr, BLOCK: tl.constexpr, I64: tl.constexpr):
    pid = tl.program_id(0)
    if I64:
        pid = pid.to(tl.int64)
        ar = tl.arange(0, BLOCK).to(tl.int64)
    else:
        ar = tl.arange(0, BLOCK)
    offs = pid * BLOCK + ar
    v = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, v)


_FLAT_COPY_BYTE_LIMIT = 12 * 1024

# Above the limit the tuned pointwise_dynamic path wins -- same config as the
# other *_copy ops (larger buffer + unroll8, vectorization kept open).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _squeeze_copy_flat(src):
    return src


def squeeze_copy(x: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``x`` with every size-1 dimension removed.

    ``aten::squeeze_copy`` (no-dim overload) never aliases its input and never
    reorders elements, so the work is: allocate ``squeezed_shape`` and copy
    ``x.numel()`` elements flat.
    """
    logger.debug("GEMS_KUNLUNXIN SQUEEZE_COPY")
    squeezed_shape = tuple(s for s in x.shape if s != 1)
    out = torch.empty(squeezed_shape, dtype=x.dtype, device=x.device, layout=x.layout)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    if x.is_contiguous():
        src = x
    else:
        src = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if x.dim() <= _MAX_FLAT_DIM:
            _launch_bcast(x.shape, x.stride(), x, src, src.numel())
        else:
            _gems_copy_(src, x)

    if n_elements * x.element_size() <= _FLAT_COPY_BYTE_LIMIT:
        block_size = _pick_flat_block(n_elements)
        _copy_flat_kernel[(triton.cdiv(n_elements, block_size),)](
            src,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            NEED_MASK=(n_elements % block_size != 0),
            num_warps=32,
            unroll_num=8,
            buffer_size_limit=1024,
        )
    else:
        _squeeze_copy_flat(src.view(-1), out0=out.view(-1))
    return out
