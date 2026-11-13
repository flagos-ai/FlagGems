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
from flag_gems.utils import libentry, tl_extra_shim

_fma = tl_extra_shim.fma

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N", "Y_AY", "Y_INV"])
def fmod_kernel(X, Y, Out, N, Y_AY, Y_INV, HAS_Y: tl.constexpr, BLOCK: tl.constexpr):
    # fmod(x, y) = copysign(fmod(|x|, |y|), x), computed as x - q*|y| with the
    # quotient q rounded-to-nearest-integer via the 2^23 magic add/sub (exact
    # trunc for |x/y| < 2^23; large quotients stay bounded by |x|*2^-24).
    # q can sit one unit on either side of trunc(|x|/|y|) because the fp32
    # quotient rounds; the two range fixups and the sign fixup below restore
    # the exact C99 remainder on the whole fp32 domain (randn-scale inputs are
    # bit-exact against the libm reference).  z == 0 covers x == +-0 (result
    # is x itself) and |y| == inf (fmod(x, inf) = x), matching ATen semantics;
    # y == 0 keeps the NaN produced by -q*0 with q = inf.  The scalar variant
    # multiplies by the host-computed reciprocal instead of dividing, so its
    # loop body is pure ALU work.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        ax = tl.abs(x)
        if HAS_Y:
            y = tl.load(Y + off, mask=mask, other=1.0).to(tl.float32)
            ay = tl.abs(y)
            z = ax / ay
        else:
            ay = Y_AY
            z = ax * Y_INV
        q = (z + 8388608.0) - 8388608.0
        r = _fma(-q, ay, ax)
        r = tl.where(r < 0.0, r + ay, r)
        r = tl.where(r >= ay, r - ay, r)
        r = tl.where(z == 0.0, x, r)
        r = tl.where(x < 0.0, -r, r)
        tl.store(Out + off, r, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 8192:
        return 2048
    if N <= 2097152:
        return 4096
    if N <= 134217728:
        return 32768
    return 32768 if is_fp32 else 65536


def _launch(inp, other, out, has_y):
    N = inp.numel()
    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    y_ptr = other if has_y else inp
    if has_y:
        ay_f = 1.0
        inv_f = 1.0
    else:
        ay_f = abs(float(other))
        inv_f = 1.0 / ay_f if ay_f != 0.0 else float("inf")
    with torch_device_fn.device(inp.device):
        fmod_kernel[(grid_size,)](
            inp,
            y_ptr,
            out,
            N,
            ay_f,
            inv_f,
            HAS_Y=has_y,
            BLOCK=BLOCK,
            num_warps=2,
        )
    return out


def _new_like(inp):
    return torch.empty_like(inp)


def fmod_scalar(A, B):
    logger.debug("GEMS_ENFLAME FMOD_SCALAR")
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, B, _new_like(inp), False)


def fmod_scalar_(A, B):
    logger.debug("GEMS_ENFLAME FMOD_SCALAR_")
    if not A.is_contiguous():
        A.copy_(fmod_scalar(A, B))
        return A
    return _launch(A, B, A, False)


def fmod_tensor(A, B):
    logger.debug("GEMS_ENFLAME FMOD_TENSOR")
    inp = A if A.is_contiguous() else A.contiguous()
    y = B if B.is_contiguous() else B.contiguous()
    return _launch(inp, y, _new_like(inp), True)


def fmod_tensor_(A, B):
    logger.debug("GEMS_ENFLAME FMOD_TENSOR_")
    y = B if B.is_contiguous() else B.contiguous()
    if not A.is_contiguous():
        A.copy_(fmod_tensor(A, y))
        return A
    return _launch(A, y, A, True)


def fmod_(A, B):
    logger.debug("GEMS_ENFLAME FMOD_")
    if isinstance(B, torch.Tensor):
        if B.dim() == 0:
            return fmod_scalar_(A, float(B))
        return fmod_tensor_(A, B)
    return fmod_scalar_(A, B)
