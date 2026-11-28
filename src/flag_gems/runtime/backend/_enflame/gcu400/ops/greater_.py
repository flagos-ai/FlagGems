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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 16
_GRID_CAP = NUM_SIPS * 3
_TINY_N = 4096
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INT32_SAFE_LIMIT = 0x7FFFFFFF

# Seed flat-kernel tier kept for non-bool non-float dtypes (see docstring).
_BLOCK = 16384


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater__inplace_func(x, y):
    return x > y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater__scalar_inplace_func(x, y):
    return x > y


@triton.jit
def _greater_inplace_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    """``a[i] = (a[i] > b[i])`` written back into ``a`` in ``a``'s own dtype.

    Kept for non-bool non-float fast-path shapes (ints): identical to the
    seed behavior apart from the host-side int32-offset guard.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    if a_ptr.dtype.element_ty == tl.float16 or a_ptr.dtype.element_ty == tl.bfloat16:
        # Compare in float32: the narrowing back to the operand type happens on
        # the store anyway, and this keeps the comparison order exact.
        r = a.to(tl.float32) > b.to(tl.float32)
    else:
        r = a > b
    tl.store(a_ptr + offs, tl.where(r, 1, 0).to(a_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def gt_inplace_flat_kernel(
    A, B, OUT, N_total, BLOCK: tl.constexpr, EXACT: tl.constexpr, NS: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=NS):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(A + off)
            y = tl.load(B + off)
            r = x > y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            mask = off < N_total
            x = tl.load(A + off, mask=mask, other=0)
            y = tl.load(B + off, mask=mask, other=0)
            r = x > y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=mask)


def _pair_config(n, itemsize):
    """Tuned in-place tiers (greater_equal_ twin; 6 B/elem 16-bit, 12 B/elem fp32)."""
    if n <= _TINY_N:
        # fp32 needs nw2 here: one warp holding 128 fp32 lanes per operand
        # serializes (4096/nw1 = 6.4us vs 3.77us at nw2); 16-bit is fine at nw1.
        block = 1 << (n - 1).bit_length() if n > 1 else 1
        return max(block, 256), (1 if itemsize == 2 else 2), 1
    if itemsize == 2:
        if n <= (1 << 17):
            return 8192, 4, 1
        if n <= (1 << 23):
            return 16384, 4, 1
        return 65536, 2, 1
    if n <= (1 << 21):
        return 8192, 4, 1
    # fp32 large tiers keep 32768/nw2 in-place: the bool-out 16384/nw1 arm
    # collapses on the 12 B/elem dtype-out stream (ge_ paired probe 2026-11-16).
    return 32768, 2, 1


def _launch(kernel, grid, args, block, num_warps, **constexprs):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; converterResult=-2 family, see P02); retrying the
    # launch recompiles and succeeds (add.py / greater.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kernel[(grid,)](
                    *args,
                    BLOCK=block,
                    num_warps=num_warps,
                    num_stages=1,
                    **constexprs,
                )
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning(
                "greater_ flat kernel launch failed (attempt %d)",
                attempt + 1,
                exc_info=True,
            )


def _pair_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    block, num_warps, num_stages = _pair_config(n, A.element_size())
    if n > _INT32_SAFE_LIMIT - block:
        return False
    grid = min(-(-n // block), _GRID_CAP)
    _launch(
        gt_inplace_flat_kernel,
        grid,
        (A, B, out, n),
        block,
        num_warps,
        EXACT=n % block == 0,
        NS=num_stages,
    )
    return True


def _seed_int_run(A, B):
    """Seed flat kernel for non-bool non-float contiguous shapes (ints)."""
    n = A.numel()
    if n > _INT32_SAFE_LIMIT - _BLOCK:
        return False
    with torch_device_fn.device(A.device):
        _greater_inplace_kernel[(-(-n // _BLOCK),)](A, B, n, BLOCK=_BLOCK, num_warps=4)
    return True


def greater_(A, B):
    logger.debug("GEMS_ENFLAME GREATER_")
    n = A.numel()
    if n == 0:
        return A
    if isinstance(B, torch.Tensor):
        if A.device != B.device:
            B = B.to(A.device)
        if (
            A.dtype == B.dtype
            and A.shape == B.shape
            and A.is_contiguous()
            and B.is_contiguous()
        ):
            if A.dtype in _FLOAT_DTYPES:
                if _pair_run(A, B, A):
                    return A
            elif A.dtype is not torch.bool:
                if _seed_int_run(A, B):
                    return A
        greater__inplace_func(A, B, out0=A)
        return A
    greater__scalar_inplace_func(A, B, out0=A)
    return A
