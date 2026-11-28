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
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import CodeGenConfig, pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SLOTS = 48

# ---------------------------------------------------------------------------
# Flat grid-stride fast path (GCU400).
#
# The pointwise_dynamic codegen with the per-op config below emits a
# tile-4096 / grid<=1024 / nw4 kernel. At the core shapes that layout pays a
# per-CTA tile-op tax the flat persistent kernel does not: treating the
# same-shape contiguous case as one dense 1D range with BLOCK=65536/nw2
# (2-byte dtypes) or 32768/nw2 (fp32) at grid<=48 measures (do_bench kernel
# median, warmup 1000ms / rep 100ms):
#   1Gi fp16  5.779 -> 2.844 ms (mem floor 2.184; 2.06 TB/s)
#   1Gi bf16  5.790 -> 2.844 ms
#   1Gi fp32  6.400 -> 4.394 ms (mem floor 4.300; 2.73 TB/s)
#   4096x4096 fp16 0.1266 -> ~0.08 ms (flat ladder); fp32 similar
# The body is a postsel form of the aten xlogy semantics:
#   prod = x * log(y) computed unguarded, then
#   out  = where((x == 0) & (y == y), 0.0, prod)
# which reproduces aten corner semantics exactly:
#   y NaN           -> cond false -> x*log(NaN) = NaN (even for x == 0)
#   x == 0, y finite-> cond true  -> 0
#   otherwise       -> x * log(y)
# Feeding log directly from the load-cast vector (instead of a where-derived
# safe_y) keeps the log on the fast fused-lowering path; the safe_y form
# measures 4.17 ms at 1Gi fp16 (where-derived vector) vs 2.84 ms here.
# Corner matrix (x in {0,-0,1,-1,3,-3,inf,-inf,nan,0.5}) x
# (y in {0,-0,1,-1,0.5,-0.75,inf,-inf,nan}) matches the CPU reference for
# fp16/bf16/fp32 (rtol/atol per dtype, equal_nan, NaN-pattern identical).
# Config ladder follows ops/add.py: keep ~256KB of load tiles per program,
# i.e. BLOCK = 131072 // itemsize at the top tier (inside the 448KB
# double-program DSM budget so grid=48 runs 48-way); tiny shapes sit on the
# launch floor with a single exact next_pow2 CTA.
# Everything the fast guard rejects (shape/broadcast mismatch, non-float or
# mixed dtypes, non-contiguous layouts) keeps the pointwise_dynamic arm
# unchanged.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _xlogy_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
    IS16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            y = tl.load(y_ptr + off, mask=mask)
        x_fp32 = x.to(tl.float32)
        y_fp32 = y.to(tl.float32)
        if IS16:
            # 16-bit fast tail: narrow the product first, then run both
            # predicates on the raw 16-bit loads and the select in the
            # 16-bit domain. Bit-identical to the postsel semantics:
            # y NaN -> cond false -> prod16 NaN (aten NaN precedence, incl.
            # x == 0); x == 0 with finite y (incl. +-inf) -> 0; the cast of
            # prod before vs after the select rounds the same fp32 value.
            prod16 = (x_fp32 * tl.log(y_fp32)).to(x.dtype)
            cond = (x == 0.0) & (y == y)
            result = tl.where(cond, 0.0, prod16)
        else:
            # fp32: postsel form (select chain fully hidden at fp32 widths).
            prod = x_fp32 * tl.log(y_fp32)
            result = tl.where((x_fp32 == 0.0) & (y_fp32 == y_fp32), 0.0, prod)
        if EXACT:
            tl.store(out_ptr + off, result)
        else:
            tl.store(out_ptr + off, result, mask=mask)


def _xlogy_choose_block_and_warps(N_total, itemsize):
    if N_total <= 4096:
        # Half-size tile, twice the CTAs: 3.74 vs 4.04 us at N=4096.
        return max(triton.next_power_of_2(N_total) // 2, 256), 2
    if N_total <= 65536:
        return 2048, 4
    if N_total <= (1 << 21):
        return 8192, 4
    if itemsize == 2 and N_total <= (1 << 25):
        # 16-bit mid band (2M < N <= 32M): B32768 measures -1.8~-2.6% at
        # the 16.7M core shapes; at 1Gi it flips to +3.7% (band capped).
        return 32768, 2
    return 131072 // itemsize, 2


def _xlogy_flat_applicable(A, B):
    return (
        isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and B.dtype == A.dtype
        and B.device == A.device
        and A.dim() == B.dim()
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    )


def _launch_xlogy_flat(A, B, out):
    N_total = A.numel()
    if out is None:
        out = torch.empty_like(A)
    if N_total == 0:
        return out
    BLOCK, num_warps = _xlogy_choose_block_and_warps(N_total, A.element_size())
    grid_size = min(-(-N_total // BLOCK), NUM_SLOTS)
    exact = N_total % BLOCK == 0
    is16 = A.element_size() == 2
    # The external llvm converter intermittently fails on cold compiles
    # (converterResult=-2 family, P02); retrying the launch recompiles and
    # succeeds (add.py / sinh.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                _xlogy_flat_kernel[(grid_size,)](
                    A,
                    B,
                    out,
                    N_total,
                    BLOCK=BLOCK,
                    EXACT=exact,
                    IS16=is16,
                    num_warps=num_warps,
                    num_stages=1,
                )
            return out
        except Exception:
            if attempt == 2:
                raise
            logger.warning("xlogy flat kernel launch failed (attempt %d)", attempt + 1)
    return out


# GCU400's default 65536-element tile with only 48 CTAs puts too much register pressure on one CTA;
# libdevice's extern __ef_logf is also much slower than Triton's core tl.log. Customize
# this operator's codegen config by measurement, leaving global CODEGEN_COFIGS untouched.
# Note: codegen_config_utils must not be imported in this module; pointwise_dynamic copies the module's
# imports verbatim into generated code, and relative imports cannot resolve under the generated gcu400.ops.* path.
_XLOGY_CONFIG = CodeGenConfig(4096, (1024, 1, 1), 4, True, prefer_1d_tile=False)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=_XLOGY_CONFIG)
@triton.jit
def special_xlogy_func(x, y):
    # xlogy(x, y) = x * log(y), but x == 0 must return 0 (for y of 0/negative/inf alike),
    # the only exception being y = NaN: then NaN is returned regardless of x.
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    y_is_nan = y_fp32 != y_fp32
    x_is_zero = x_fp32 == 0.0
    # Replace "x is 0 and y is not NaN" inputs with 1: log(1) = 0, so the product is naturally 0;
    # this removes two trailing tl.where layers and avoids log's slow exceptional branch for 0/negative values.
    # Keep NaN when y is NaN; 0 * NaN is still NaN, semantics unchanged.
    safe_y = tl.where(x_is_zero & ~y_is_nan, 1.0, y_fp32)
    return x_fp32 * tl.log(safe_y)


def special_xlogy(A, B):
    logger.debug("GEMS_ENFLAME SPECIAL_XLOGY")
    if _xlogy_flat_applicable(A, B):
        return _launch_xlogy_flat(A, B, None)
    return special_xlogy_func(A, B)


def special_xlogy_(A, B):
    logger.debug("GEMS_ENFLAME SPECIAL_XLOGY_")
    if _xlogy_flat_applicable(A, B):
        return _launch_xlogy_flat(A, B, A)
    return special_xlogy_func(A, B, out0=A)
