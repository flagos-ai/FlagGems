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

import flag_gems
from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b

    return result


@triton.jit
def _glu_flat(
    X,
    Y,
    N,
    HALF: tl.constexpr,
    FULL: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
):
    """Flat layout (default path): reads a/b straight from the contiguous tensor, skipping chunk copies.

    Each output element derives (row, col) via `off // HALF` and `off % HALF`; with HALF a
    compile-time constant (mostly a power of two) both become shifts/masks. Measured optimal
    for HALF >= 256 or fp32 (4K tile, 48 CTA, 4 warp); only for 2-byte dtypes with HALF <= 128
    does it lose to the 2D tile, see `_glu_tile`.
    """
    pid = tl.program_id(0)
    for base in range(pid * BLOCK, N, STRIDE * BLOCK):
        off = base + tl.arange(0, BLOCK)
        m = off < N
        outer = off // HALF
        # Must use % instead of off - outer*HALF: GCU's integer division goes through a reciprocal approximation, so
        # some indices
        # can be off by 1; using subtraction amplifies the error into whole-row misalignment (measured on (1024,1024),
        # (.,.,.,60) with
        # 50% of elements wrong), whereas the lowering of % is exact.
        inner = off % HALF
        a = tl.load(X + outer * FULL + inner, mask=m, other=0)
        b = tl.load(X + outer * FULL + HALF + inner, mask=m, other=0)
        sig = tl.fdiv(1.0, 1.0 + tl.exp(-b.to(tl.float32)))
        tl.store(Y + off, (a.to(tl.float32) * sig).to(Y.dtype.element_ty), mask=m)


@triton.jit
def _glu_tile(
    X,
    Y,
    M,
    HALF: tl.constexpr,
    FULL: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_PROGS: tl.constexpr,
    MASK_ROW: tl.constexpr,
    EVEN_COL: tl.constexpr,
):
    """Row block plus column vectorization: for 2-byte dtypes with HALF <= 128.

    There the flat layout's a/b reads are short 128 x 2-byte gathers measuring only 1.0~1.2x of
    torch; a (ROWS x COLS) = (128 x 128) 2D block read contiguously by rows lifts the same shape
    to 2.7x. The column mask is dropped when COLS divides HALF (EVEN_COL).
    """
    pid = tl.program_id(0)
    rr = tl.arange(0, ROWS)[:, None]
    for row0 in tl.range(pid * ROWS, M, ROW_PROGS * ROWS):
        rows = row0 + rr
        if MASK_ROW:
            rmask = rows < M
        in_base = rows * FULL
        out_base = rows * HALF
        for cbase in tl.range(0, HALF, COLS):
            cc = cbase + tl.arange(0, COLS)[None, :]
            if EVEN_COL:
                if MASK_ROW:
                    m = rmask
                    a = tl.load(X + in_base + cc, mask=m, other=0.0).to(tl.float32)
                    b = tl.load(X + in_base + HALF + cc, mask=m, other=0.0).to(
                        tl.float32
                    )
                    sig = tl.fdiv(1.0, 1.0 + tl.exp(-b))
                    tl.store(Y + out_base + cc, a * sig, mask=m)
                else:
                    a = tl.load(X + in_base + cc).to(tl.float32)
                    b = tl.load(X + in_base + HALF + cc).to(tl.float32)
                    sig = tl.fdiv(1.0, 1.0 + tl.exp(-b))
                    tl.store(Y + out_base + cc, a * sig)
            else:
                if MASK_ROW:
                    m = (cc < HALF) & rmask
                else:
                    m = cc < HALF
                a = tl.load(X + in_base + cc, mask=m, other=0.0).to(tl.float32)
                b = tl.load(X + in_base + HALF + cc, mask=m, other=0.0).to(tl.float32)
                sig = tl.fdiv(1.0, 1.0 + tl.exp(-b))
                tl.store(Y + out_base + cc, a * sig, mask=m)


_GLU_GRID = 48
_GLU_BLOCK = 4096
_GLU_TILE = 16384
_GLU_NARROW_HALF = 128


def _glu_launch(inp, out, half, full):
    m = inp.numel() // full
    n = m * half
    if inp.element_size() <= 2 and half <= _GLU_NARROW_HALF:
        # 2-byte + narrow columns: a 2D tile clearly beats the flat one (2.7x vs 1.0~1.2x)
        cols = max(64, triton.next_power_of_2(half))
        rows = max(1, _GLU_TILE // cols)
        row_progs = max(1, min(_GLU_GRID, triton.cdiv(m, rows)))
        _glu_tile[(row_progs,)](
            inp,
            out,
            m,
            half,
            full,
            rows,
            cols,
            row_progs,
            (m % rows) != 0,
            (half % cols) == 0,
            num_warps=4,
        )
        return
    _glu_flat[(min(_GLU_GRID, triton.cdiv(n, _GLU_BLOCK)),)](
        inp, out, n, half, full, _GLU_BLOCK, _GLU_GRID, num_warps=4
    )


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS_ENFLAME GLU")
    last_dim = self.dim() - 1
    if (dim == -1 or dim == last_dim) and self.is_contiguous():
        # t8 batch Step 4: the original implementation's torch.chunk + two .contiguous() full copies
        # + codegen wrapper kept all 12 cases at a constant ~0.05ms; a single kernel reads a/b directly
        # by stride.
        full = self.shape[-1]
        half = full // 2
        out = torch.empty(
            self.shape[:-1] + (half,), dtype=self.dtype, device=self.device
        )
        _glu_launch(self, out, half, full)
        return out
    # Fall back to the original implementation for non-last-dim dim / non-contiguous inputs to keep semantics unchanged
    a, b = torch.chunk(self, 2, dim=dim)
    a = a.contiguous()
    b = b.contiguous()
    return glu_kernel(a, b)


@pointwise_dynamic(
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ]
)
@triton.jit
def glu_backward_kernel(grad_output, a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    da = grad_output * sigmoid_b
    db = grad_output.to(tl.float32) * a * sigmoid_b * (1.0 - sigmoid_b)

    return da, db


def glu_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS_ENFLAME GLU_BACKWARD")
    # Split into a and b, then materialize them contiguously: the enflame
    # block-pointer pointwise path cannot handle the non-contiguous views that
    # torch.chunk produces along a non-outer dim.
    a, b = torch.chunk(self, 2, dim=dim)
    a = a.contiguous()
    b = b.contiguous()
    grad_output = grad_output.contiguous()
    grad_a, grad_b = glu_backward_kernel(grad_output, a, b)
    grad_input = flag_gems.cat((grad_a, grad_b), dim=dim)

    return grad_input
