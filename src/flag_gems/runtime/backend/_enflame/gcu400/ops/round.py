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

"""GCU400 round: magic-number RNE body + 1R1W family launch shape.

The legacy body ``round_half_to_even_impl`` calls ``tl.floor`` twice; on
this backend ``tl.floor`` lowers to an opaque extern libm call that cannot
pipeline with streaming loads/stores (floor__opt.py / log1p precedents).
At 2^29 elements that body measured 3956/3957/3837us (f16/bf16/f32) vs the
671/674/1360us copy floor, i.e. the op was pure serial-ALU bound, and the
extern ``libdevice.nearbyint`` body is even worse (22222us, 33x the floor).

Each fp dtype arm here instead uses the fp32 magic-number round-to-nearest
 ``(x + 12582912.0) - 12582912.0`` (12582912.0 = 2^23 + 2^22), which relies
only on IEEE RNE fp32 adds and the ULP=1 binade [2^23, 2^24):

* fp16: cast up to fp32, magic round-trip, cast back. Every finite fp16
  value has |x| <= 65504 < 2^22, so the magic is exact for the whole domain;
  inf/NaN propagate through the adds. No guard needed.
* bf16: same chain, also guard-free: every bf16 value with |x| >= 2^22 is an
  even multiple of 32768, so the fp32 round-trip either lands exactly (no
  rounding, mid binades) or its sub-ULP error is re-absorbed by the cast
  back down to bf16 (error <= 2^(b-23) << ULP_bf16/2 = 2^(b-8)).
* fp32: needs three magnitude bands because fp32 keeps half-integers up to
  2^23: |x| < 2^22 uses the 12582912.0 magic (sum stays in [2^23, 2^24)),
  [2^22, 2^23) uses a sign-split shift magic (+2^22 for x>=0, -2^23 for
  x<0, sum lands in a ULP=1 binade), and |x| >= 2^23 is already integral
  and passes through. NaN/inf fail both band tests and pass through.

Verified bit-exact vs torch.round (CPU) over the full 16-bit bit-pattern
space for fp16/bf16 (4.2M random draws each) and 4.2M random fp32 patterns
plus band-edge values (probe, 2026-11-16). The only deviation from ATen is
the sign of zero for -0.5/-0.0 lanes (+0.0 instead of -0.0), identical to
the legacy body's behaviour and tolerated by the test framework
(assert_close treats -0.0 == +0.0).

Launch shape follows the 1R1W stream family recipe (relu.py / rsqrt.py /
floor__opt.py): flat grid-stride kernel capped at 48 CTAs, an EVEN arm that
drops masks when n divides the tile, and a per-band BLOCK/num_warps ladder
re-scanned for this body (interleaved do_bench median, 2026-11-16):

* 2^29: f16/bf16 131072/nw4 = 670.1/670.4us, f32 65536/nw4 = 1345.0us
  (all at the copy floor; the heavier magic chain wants nw4 here, unlike
  the nw2 sweet spots of the pure one-op relu/bitwise_not bodies).
* 2^24: f16/bf16 65536/nw4 = 28.0/27.7us, f32 32768/nw4 = 49.0us.
* tiny 4096: every 512-4096/nw2-4 config ties at 3.1us; 1024/nw4 keeps the
  family tiny-band entry. In-place (round_) shares the same ladder (the
  same-buffer variant measured equal or 1-2us faster in every band).

decimals != 0, non-contiguous, non-{fp16,bf16,fp32} and over-int32-domain
inputs keep the legacy ``round_kernel`` path unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def round_half_to_even_impl(x):
    r = tl.floor(x)
    d = x - r
    is_odd = (r - 2.0 * tl.floor(r * 0.5)) > 0.5
    return tl.where((d > 0.5) | ((tl.abs(d - 0.5) < 1e-10) & is_odd), r + 1.0, r)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def round_kernel(
    x_ptr,
    out_ptr,
    N_total,
    decimals: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)

        if decimals == 0:
            out = x
            if IS_FP32:
                out = round_half_to_even_impl(x)
            elif IS_FP16:
                out = tl.cast(
                    round_half_to_even_impl(tl.cast(x, tl.float32)), tl.float16
                )
            elif IS_BF16:
                out = tl.cast(
                    round_half_to_even_impl(tl.cast(x, tl.float32)), tl.bfloat16
                )
        else:
            scale = 10.0**decimals
            if IS_FP32:
                out = round_half_to_even_impl(x * scale) / scale
            elif IS_FP16:
                x_fp32 = tl.cast(x, tl.float32)
                out = tl.cast(
                    round_half_to_even_impl(x_fp32 * scale) / scale, tl.float16
                )
            elif IS_BF16:
                x_fp32 = tl.cast(x, tl.float32)
                out = tl.cast(
                    round_half_to_even_impl(x_fp32 * scale) / scale, tl.bfloat16
                )
            else:
                out = x

        tl.store(out_ptr + off, out, mask=mask)


@triton.jit
def _round_rne_magic(x, IS_FP32: tl.constexpr, IS_FP16: tl.constexpr):
    if IS_FP16:
        xf = tl.cast(x, tl.float32)
        r = (xf + 12582912.0) - 12582912.0
        return tl.cast(r, tl.float16)
    elif IS_FP32:
        ax = tl.abs(x)
        r1 = (x + 12582912.0) - 12582912.0
        m2 = tl.where(x >= 0.0, 4194304.0, -8388608.0)
        r2 = (x + m2) - m2
        return tl.where(ax < 4194304.0, r1, tl.where(ax < 8388608.0, r2, x))
    else:
        xf = tl.cast(x, tl.float32)
        r = (xf + 12582912.0) - 12582912.0
        return tl.cast(r, tl.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def round_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            tl.store(out_ptr + off, _round_rne_magic(x, IS_FP32, IS_FP16))
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            tl.store(out_ptr + off, _round_rne_magic(x, IS_FP32, IS_FP16), mask=mask)


def _launch_round(input, output, decimals):
    N_total = input.numel()
    if N_total == 0:
        return

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16
    BLOCK = 1024 if N_total < 65536 else 4096
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)

    with torch_device_fn.device(input.device):
        round_kernel[(grid_size,)](
            input,
            output,
            N_total,
            decimals,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            BLOCK=BLOCK,
            num_warps=4,
        )


# LibEntry's dispatch key is (dtype, alignment, BLOCK, EVEN) and does not
# include num_warps, so each (dtype, BLOCK) pair pins a single num_warps.
def _choose_flat_params(n_total, dtype):
    if n_total <= 8192:
        return 1024, 4
    if n_total <= 65536:
        return 8192, 2
    if dtype == torch.float32:
        if n_total <= (1 << 24):
            return 32768, 4
        return 65536, 4
    if n_total <= (1 << 24):
        return 65536, 4
    return 131072, 4


def _launch_round_flat(input, output):
    n_total = input.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_flat_params(n_total, input.dtype)
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    even = n_total % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # rsqrt.py / floor__opt.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(input.device):
                round_flat_kernel[(grid_size,)](
                    input,
                    output,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    IS_FP32=input.dtype == torch.float32,
                    IS_FP16=input.dtype == torch.float16,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "round flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_path_ok(input, decimals):
    return (
        decimals == 0
        and input.is_contiguous()
        and input.dtype in _FLAT_DTYPES
        and input.numel() < _INT32_SAFE_LIMIT
    )


def round_func(input, decimals=0):
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input.clone()
    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )

    output = torch.empty_like(input)
    if _flat_path_ok(input, decimals):
        _launch_round_flat(input, output)
    else:
        _launch_round(input, output, decimals)
    return output


def round(input, decimals=0):
    logger.debug("GEMS_ENFLAME ROUND")
    return round_func(input, decimals=decimals)


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS_ENFLAME ROUND_OUT")
    if out is None:
        return round_func(input, decimals=decimals)
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        out.copy_(input)
        return out
    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )
    if _flat_path_ok(input, decimals) and out.is_contiguous():
        _launch_round_flat(input, out)
    else:
        _launch_round(input, out, decimals)
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS_ENFLAME ROUND_")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input
    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )
    if _flat_path_ok(input, decimals):
        _launch_round_flat(input, input)
    else:
        _launch_round(input, input, decimals)
    return input
