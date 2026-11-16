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

logger = logging.getLogger(__name__)


_TL_DTYPES = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
}


@triton.jit
def _scalar_fill_kernel(out_ptr, value, DTYPE: tl.constexpr):
    tl.store(out_ptr, tl.full((), value, DTYPE))


@triton.jit
def _scalar_fill_f64_kernel(out_ptr, value: tl.constexpr):
    tl.store(out_ptr, tl.full((), value, tl.float64))


def scalar_tensor(s, *, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        if isinstance(s, bool):
            dtype = torch.bool
        elif isinstance(s, int):
            dtype = torch.int64
        elif isinstance(s, float):
            dtype = torch.float32
        else:
            dtype = torch.float32

    if layout is None:
        layout = torch.strided

    out = torch.empty(
        (), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
    )

    if dtype == torch.bool:
        _scalar_fill_kernel[(1,)](out, bool(s), tl.int1)
    elif dtype == torch.float64:
        _scalar_fill_f64_kernel[(1,)](out, float(s))
    elif dtype.is_floating_point:
        _scalar_fill_kernel[(1,)](out, float(s), _TL_DTYPES[dtype])
    else:
        _scalar_fill_kernel[(1,)](out, int(s), _TL_DTYPES[dtype])

    return out
