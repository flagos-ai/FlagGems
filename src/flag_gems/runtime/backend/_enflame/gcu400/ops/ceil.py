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

"""GCU400 ceil: int-domain rounding bodies + 1R1W family launch shape.

The seeded body lowered ``tl.ceil(x.to(tl.float32))`` to the extern software
path (``tops_ceil_v2_fp32``, the floor/rint extern family; llir-verified)
behind a stale B65536 ladder with grid capped at 24 CTAs, which kept every
core case at 0.88-0.99x native (baseline 2026-11-16: fp16/bf16 512M 1599us =
1.34TB/s effective, fp32 512M 1527us).  Each dtype arm here replaces it with
the int-domain trunc chain (the ``floor__opt`` / ``_foreach_unary._ceil_func``
recipe with the correction direction flipped: trunc via an int32 round-trip in
the *storage* domain plus ``+1`` when truncation moved down; probe-verified
value-exact against torch.ceil over specials incl. +-inf/NaN/subnormals and
the 2^23/2^24/2^31 boundaries; the only deviation is the sign of a -0.0
result, which torch.testing.assert_close(atol=0, rtol=0) accepts):

* fp16: every finite fp16 fits int32, so only inf/NaN need the
  ``x - x == 0`` exclusion;
* bf16/fp32: finite values beyond int32 exist and are already integral, so
  lanes with ``|x| >= 2^31`` (inf/NaN included) pass the input through.

The extern body measured +781us at 512M fp16 and +80us at 512M fp32 over
this chain (interleaved do_bench median, probe2 2026-11-16); the int body
tracks the floor__opt sibling within 0.5-3% and beats native ceil on every
core case (best 0.485x at fp16 512M).

Launch shape follows the 1R1W stream family recipe (abs.py / floor__opt.py
retunes): flat grid-stride kernel capped at 48 CTAs, an EVEN arm that drops
masks when n divides the tile, and a per-band BLOCK/num_warps ladder
re-scanned for this body across probe2-probe4 (interleaved do_bench median,
2026-11-16; head-to-head 4-round run resolved a cross-session board-state
flip on the 16-bit big bucket in favor of B65536/nw4):  tiny wants
B1024/nw4, fp32 tracks B65536/nw4 from 64K up (mid configs flat, big beats
B32768 by ~0.5%), fp16 wants B65536/nw4 everywhere past 64K (B131072 costs
+1.9% at 512M), bf16 wants B32768/nw4 up to 2^24 and B65536/nw4 beyond.
In-place and out-of-place sweet points coincide on every band, so one
ladder serves ceil / ceil_ / ceil_out without colliding on the LibEntry
dispatch key (key = dtype, alignment, BLOCK, EVEN; num_warps not included).
Non-contiguous, non-{fp16,bf16,fp32} and oversized inputs keep the generic
pointwise path; ceil_ additionally short-circuits integral/bool inputs
(identity) the way floor_ does.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 12

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def ceil_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x0 = tl.load(x_ptr + off)
            if IS_FP16:
                trunc = x0.to(tl.int32).to(tl.float16)
                rounded = tl.where(trunc < x0, trunc + 1.0, trunc)
                out = tl.where(x0 - x0 == 0, rounded, x0)
            elif IS_BF16:
                trunc = x0.to(tl.int32).to(tl.bfloat16)
                rounded = tl.where(trunc < x0, trunc + 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            else:
                trunc = x0.to(tl.int32).to(tl.float32)
                rounded = tl.where(trunc < x0, trunc + 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            tl.store(out_ptr + off, out)
        else:
            mask = off < N_total
            x0 = tl.load(x_ptr + off, mask=mask)
            if IS_FP16:
                trunc = x0.to(tl.int32).to(tl.float16)
                rounded = tl.where(trunc < x0, trunc + 1.0, trunc)
                out = tl.where(x0 - x0 == 0, rounded, x0)
            elif IS_BF16:
                trunc = x0.to(tl.int32).to(tl.bfloat16)
                rounded = tl.where(trunc < x0, trunc + 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            else:
                trunc = x0.to(tl.int32).to(tl.float32)
                rounded = tl.where(trunc < x0, trunc + 1.0, trunc)
                out = tl.where(tl.abs(x0) < 2147483648.0, rounded, x0)
            tl.store(out_ptr + off, out, mask=mask)


# One ladder serves out-of-place and in-place (their sweet points coincide,
# probe3/probe4), and each (dtype, BLOCK) pair pins a single num_warps so
# the shared LibEntry dispatch key never collides across entry points.
def _choose_launch(n_total, dtype):
    if n_total <= 8192:
        return 1024, 4
    if n_total <= 65536:
        return 8192, 2
    if dtype == torch.bfloat16 and n_total <= (1 << 24):
        return 32768, 4
    return 65536, 4


def _launch_ceil(inp, out):
    n_total = inp.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, inp.dtype)
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    even = n_total % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # floor__opt.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                ceil_flat_kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    IS_FP16=inp.dtype == torch.float16,
                    IS_BF16=inp.dtype == torch.bfloat16,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "ceil flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_path_ok(t):
    return (
        t.is_contiguous() and t.dtype in _FLAT_DTYPES and t.numel() < _INT32_SAFE_LIMIT
    )


def ceil(A):
    logger.debug("GEMS_ENFLAME CEIL")
    if not _flat_path_ok(A):
        from flag_gems.ops.ceil import ceil_func as _public

        return _public(A)
    out = torch.empty_like(A)
    _launch_ceil(A, out)
    return out


def ceil_(A):
    logger.debug("GEMS_ENFLAME CEIL_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("ceil_ expects a torch.Tensor.")
    if A.is_complex():
        raise TypeError("ceil_ is not supported for complex tensors.")

    if A.dtype not in _FLAT_DTYPES:
        # ceil is the identity on integral/bool tensors; fp64 is
        # unsupported on gcu400 (floor_ seed-class behavior).
        return A

    n_total = A.numel()
    if n_total == 0:
        return A

    if not A.is_contiguous() or n_total >= _INT32_SAFE_LIMIT:
        from flag_gems.ops.ceil import ceil_func as _public

        _public(A, out0=A)
        return A

    _launch_ceil(A, A)
    return A


def ceil_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME CEIL_OUT")
    if out is None:
        return ceil(A)
    if (
        not _flat_path_ok(A)
        or not out.is_contiguous()
        or out.dtype != A.dtype
        or out.numel() != A.numel()
    ):
        from flag_gems.ops.ceil import ceil_func as _public

        _public(A, out0=out)
        return out
    _launch_ceil(A, out)
    return out
