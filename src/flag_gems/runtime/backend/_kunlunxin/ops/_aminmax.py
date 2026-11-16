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

"""Kunlunxin override for ``aten::_aminmax`` / ``aten::_aminmax.out``.

The generic FlagGems implementation (``flag_gems/ops/_aminmax.py``) loads the
same pointer tile twice with *different* masked-load ``other`` values
(``other=min_fill`` for the min bucket and ``other=max_fill`` for the max
bucket).  On XPU that dual-``other`` masked load is miscompiled: both loads end
up using the *min* fill.  When the tail lane is masked the max accumulator then
reads ``+inf`` (or ``INT_MAX`` for int32), so a shape whose ``numel`` is not a
multiple of the stage-1 block (e.g. ``(200, 40999, 3)`` -> BLOCK=8192,
residue=7016) returns ``inf``/``INT_MAX`` for the max output.

Fix: the stage-1 reduction is split into a full-block pass and a lone residue
pass.  The full-block pass keeps the generic dual-masked-load reducer verbatim
(its `mask` is all-true there, so the colliding `other` values are never used
and the fast path is byte-identical to the generic kernel).  The ``M % BLOCK``
residue, which is what triggered the miscompile, is handed to a separate
one-program ``_aminmax_kernel_tail`` that does a single ``other=0.0`` masked
load and selects the per-bucket identity with ``tl.where`` (the verified
Kunlunxin ``aminmax_kernel_1`` shape, correct at BLOCK=8192).  The rest of the
algorithm (sqrt-balanced two-stage reduction, NaN propagation, ``.out``
validation) mirrors the generic implementation, so throughput is unchanged.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _aminmax_kernel_1(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    # Full-block program. `M` is always a multiple of BLOCK_SIZE here, so the
    # `mask` is all-true and the two loads below never take their `other`
    # value: the XPU miscompile (same address, two different `other` values ->
    # CSE keeps one fill) is therefore harmless, and this stays byte-for-byte
    # the generic reducer's fast path. The masked residue is handled by
    # `_aminmax_kernel_tail` instead.
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_fill = get_dtype_max(inp.type.element_ty)
    max_fill = get_dtype_min(inp.type.element_ty)
    min_val = tl.load(inp_ptrs, mask=mask, other=min_fill)
    max_val = tl.load(inp_ptrs, mask=mask, other=max_fill)

    if IS_FLOAT:
        # torch._aminmax returns NaN for *both* outputs when any input is NaN,
        # but tl.min/tl.max drop NaNs instead of propagating them. Detect a NaN
        # in this tile (x != x is only true for NaN) and force both results to
        # NaN, which also keeps the second stage's inputs NaN-tainted.
        has_nan = tl.sum((min_val != min_val).to(tl.int32)) > 0
        min_red = tl.min(min_val)
        max_red = tl.max(max_val)
        nan = float("nan")
        min_val = tl.where(has_nan, nan, min_red)
        max_val = tl.where(has_nan, nan, max_red)
    else:
        min_val = tl.min(min_val)
        max_val = tl.max(max_val)

    min_ptr = min_out + pid
    max_ptr = max_out + pid
    tl.store(min_ptr, min_val)
    tl.store(max_ptr, max_val)


@libentry()
@triton.jit
def _aminmax_kernel_tail(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    # Single-program residue reducer (`M < BLOCK_SIZE`, possibly not a power of
    # two). A single masked load with one `other=0.0`, then the per-bucket
    # identity via `tl.where`, is the same shape the verified Kunlunxin
    # `aminmax_kernel_1` uses and is correct at BLOCK=8192.
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    value = tl.load(inp + offset, mask=mask, other=0.0)
    min_val = tl.where(mask, value, get_dtype_max(inp.type.element_ty))
    max_val = tl.where(mask, value, get_dtype_min(inp.type.element_ty))

    if IS_FLOAT:
        has_nan = tl.sum((value != value).to(tl.int32)) > 0
        min_red = tl.min(min_val)
        max_red = tl.max(max_val)
        nan = float("nan")
        min_val = tl.where(has_nan, nan, min_red)
        max_val = tl.where(has_nan, nan, max_red)
    else:
        min_val = tl.min(min_val)
        max_val = tl.max(max_val)

    tl.store(min_out, min_val)
    tl.store(max_out, max_val)


@libentry()
@triton.jit
def _aminmax_kernel_2(
    min_inp,
    max_inp,
    min_out,
    max_out,
    mid_size,
    BLOCK_MID: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    min_ptrs = min_inp + offset
    max_ptrs = max_inp + offset
    mask = offset < mid_size
    min_fill = get_dtype_max(min_inp.type.element_ty)
    max_fill = get_dtype_min(max_inp.type.element_ty)
    min_raw = tl.load(min_ptrs, mask=mask, other=0.0)
    max_raw = tl.load(max_ptrs, mask=mask, other=0.0)
    min_val = tl.where(mask, min_raw, min_fill)
    max_val = tl.where(mask, max_raw, max_fill)

    if IS_FLOAT:
        # Second stage needs the same guard: stage one writes NaN into both
        # partial buffers when it sees one, and tl.min/tl.max would drop it here.
        # Either buffer carrying a NaN taints both outputs.
        has_nan = (
            tl.sum((min_raw != min_raw).to(tl.int32))
            + tl.sum((max_raw != max_raw).to(tl.int32))
        ) > 0
        min_red = tl.min(min_val)
        max_red = tl.max(max_val)
        nan = float("nan")
        min_val = tl.where(has_nan, nan, min_red)
        max_val = tl.where(has_nan, nan, max_red)
    else:
        min_val = tl.min(min_val)
        max_val = tl.max(max_val)

    tl.store(min_out, min_val)
    tl.store(max_out, max_val)


def _aminmax_run(inp, out0=None, out1=None):
    """Shared driver for the base and ``.out`` variants."""
    if inp.numel() == 0:
        # ATen refuses to reduce an empty input rather than returning +-inf.
        raise RuntimeError(
            "aminmax(): cannot compute aminmax over an empty dimension as the "
            "operation has no identity."
        )

    # The kernels index with a flat offset, so a sliced, transposed or expanded
    # view would be read in the wrong element order (or past its end for an
    # expanded view, whose stride is 0). Materialize a dense copy first.
    inp = inp.contiguous()

    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    full = M // block_size
    res = M - full * block_size
    mid_size = full + (1 if res else 0)
    block_mid = triton.next_power_of_2(mid_size)
    dtype = inp.dtype
    is_float = inp.is_floating_point()

    if out0 is None:
        min_out = torch.empty([], dtype=dtype, device=inp.device)
        max_out = torch.empty([], dtype=dtype, device=inp.device)
    else:
        # ATen validates dtype and device but resizes a mis-shaped output to a
        # scalar in place, keeping the caller's objects so the returned tensors
        # alias out0/out1.
        for name, o in (("out0", out0), ("out1", out1)):
            if o.dtype != dtype:
                raise RuntimeError(
                    f"Expected out tensor to have dtype {dtype}, but got "
                    f"{o.dtype} instead"
                )
            if o.device != inp.device:
                raise RuntimeError(
                    f"Expected out tensor to have device {inp.device}, but got "
                    f"{o.device} instead"
                )
        if out0.shape != torch.Size([]):
            out0.resize_(())
        if out1.shape != torch.Size([]):
            out1.resize_(())
        min_out, max_out = out0, out1

    flat = inp.view(-1)
    min_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    max_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if full:
            _aminmax_kernel_1[(full, 1)](
                flat,
                min_mid,
                max_mid,
                full * block_size,
                block_size,
                IS_FLOAT=is_float,
            )
        if res:
            # Tail residue: one program, masked load + per-bucket identity.
            tail_block = triton.next_power_of_2(res)
            _aminmax_kernel_tail[(1, 1)](
                flat[full * block_size :],
                min_mid[full:],
                max_mid[full:],
                res,
                tail_block,
                IS_FLOAT=is_float,
            )
        _aminmax_kernel_2[(1, 1)](
            min_mid,
            max_mid,
            min_out,
            max_out,
            mid_size,
            block_mid,
            IS_FLOAT=is_float,
        )
    return min_out, max_out


def _aminmax(inp):
    logger.debug("GEMS_KUNLUNXIN _AMINMAX")
    return _aminmax_run(inp)


def _aminmax_out(inp, *, out0, out1):
    logger.debug("GEMS_KUNLUNXIN _AMINMAX_OUT")
    return _aminmax_run(inp, out0=out0, out1=out1)
