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
import struct
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# GCU400 flat-kernel override of the pointwise_dynamic path.  The generated
# block-pointer kernel (tile 65536 / grid <= 48 / num_warps 4, boundary_check
# on every load and store) reaches only ~1.5 TB/s on 16M-element shapes
# (0.70-0.81x vs torch native) and ~2.7 TB/s on 1G shapes (0.85-0.90x),
# while the plain flat grid-stride kernel below sits on the memory floor:
#   16M fp16 36.9 -> 28.2us (1.06x), 16M fp32 51.4 -> 47.6us (1.02x),
#   1G  fp16 1508 -> 1348us (~1.0x),  1G  fp32 2679 -> 2676us (~1.0x)
# (L600 probe, kernel mode do_bench median w1000/r100).  Math is identical to
# the pointwise kernel (fp32-upcast compare, bit-exact vs the CPU reference on
# a 16M-element fp16 sample), so accuracy is unchanged.  Small / non-contiguous
# / non-float / huge (> int32-safe) inputs keep the generic pointwise path,
# which is fastest below ~8K elements.
NUM_SIPS = 24
_SMALL_N = 8192
_SINGLE_CTA_N = 32768
_MID_N = 1 << 21
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@lru_cache(maxsize=1024)
def _f16_predicate_exact(t):
    # True iff fp16(fp32(t)) == fp32(t) as reals: then the fp16-native compare
    # is bit-identical to the fp32-upcast compare (no threshold rounding band).
    # Copied from threshold_.py instead of imported: pointwise_dynamic's
    # codegen copies this file's import block into its cached modules, where
    # relative imports cannot resolve.
    try:
        t32 = struct.unpack("f", struct.pack("f", t))[0]
    except OverflowError:
        return False
    if t32 != t32:  # NaN threshold: keep the fp32-upcast compare
        return False
    return float(np.float16(t32)) == t32


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def threshold_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    threshold,
    value,
    BLOCK: tl.constexpr,
    NATIVE_PRED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        m = off < N_total
        x = tl.load(x_ptr + off, mask=m)
        if NATIVE_PRED:
            # fp16 + exactly representable threshold: native-width compare
            # agrees with the fp32-upcast compare on every value (no rounding
            # band), and NaN still falls through to x.
            y = tl.where(x <= threshold.to(x.dtype), value.to(x.dtype), x)
        else:
            # aten semantics `x <= t ? value : x` with the fp32-upcast compare
            # (NaN -> NaN); the select itself runs in the element dtype
            # (bit-identical to the fp32-select form).
            cond = x.to(tl.float32) <= threshold
            y = tl.where(cond, value.to(x.dtype), x)
        tl.store(out_ptr + off, y, mask=m)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def threshold_kernel(self, threshold, value):
    return tl.where(self > threshold, self, value)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    return tl.where(self > threshold, grad_output, 0)


def _choose_block(N_total):
    # do_bench(median) sweep on GCU400: mid shapes peak at BLOCK 32768 (more
    # CTAs in flight), the 16M-1G band gains ~1.5-3% from 65536; tiny shapes
    # use one exact next_pow2 tile.  131072 was rejected: no gain over 65536
    # and it exceeds the 917504B DSM budget (rad2deg / greater_equal agree).
    if N_total <= _SINGLE_CTA_N:
        return triton.next_power_of_2(N_total)
    if N_total <= _MID_N:
        return 32768
    return 65536


def _launch_flat(x, out, threshold, value):
    N_total = x.numel()
    BLOCK = _choose_block(N_total)
    grid_size = min(triton.cdiv(N_total, BLOCK), NUM_SIPS * 2)
    num_warps = 4 if x.dtype == torch.float32 else 2
    native_pred = x.dtype == torch.float16 and _f16_predicate_exact(threshold)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (rad2deg / sinh / atan2_
    # precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                threshold_flat_kernel[(grid_size,)](
                    x,
                    out,
                    N_total,
                    threshold,
                    value,
                    BLOCK=BLOCK,
                    NATIVE_PRED=native_pred,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "threshold flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def threshold(self, threshold, value):
    logger.debug("GEMS_ENFLAME THRESHOLD")
    if (
        self.is_contiguous()
        and self.dtype in _FLOAT_DTYPES
        and _SMALL_N < self.numel() <= _INT32_SAFE_LIMIT - 65536
    ):
        out = torch.empty_like(self)
        _launch_flat(self, out, float(threshold), float(value))
        return out
    return threshold_kernel(self, threshold, value)


def threshold_backward(grad_output, self, threshold):
    logger.debug("GEMS_ENFLAME THRESHOLD_BACKWARD")
    grad_input = threshold_backward_kernel(grad_output, self, threshold)
    return grad_input
