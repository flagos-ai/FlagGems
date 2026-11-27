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
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# Flat-grid fast path is only sound while every offset stays in the int32
# domain (see the numel guard in `_fast_path_eligible`).
_MAX_FLAT_NUMEL = 2**31 - 1

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N"])
def logaddexp2_kernel(X, Y, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + off).to(tl.float32)
            y = tl.load(Y + off).to(tl.float32)
        else:
            mask = off < N
            x = tl.load(X + off, mask=mask).to(tl.float32)
            y = tl.load(Y + off, mask=mask).to(tl.float32)
        m = tl.maximum(x, y)
        delta = x - y
        # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y).
        # Evaluate via the native-precision exp/log pair (exp2/log2 lower to
        # slow paths on this backend): 2**(-|d|) = exp(-|d| * ln2) and
        # log2(u) = log(u) * log2(e).
        res = (
            m
            + tl.log(1.0 + tl.exp(-tl.abs(delta) * 0.6931471805599453))
            * 1.4426950408889634
        )
        # `delta` is NaN when x and y are equal infinities (inf - inf); the result
        # is then m, e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
        res = tl.where(delta != delta, m, res)
        # Genuine NaN inputs must still propagate NaN.
        is_nan = (x != x) | (y != y)
        res = tl.where(is_nan, float("nan"), res)
        if EVEN:
            tl.store(Out + off, res)
        else:
            tl.store(Out + off, res, mask=mask)


@pointwise_dynamic(
    config=CodeGenConfig(4096, (12, 1, 1), 8, True, False, False),
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def logaddexp2_func(x, y):
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # `delta` is NaN when x and y are equal infinities (inf - inf); the result
    # is then m, e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


def _choose_block(N, is_fp32):
    # Transcendental-dense 3-stream pointwise ladder (two loads + one store,
    # one exp + one log per element): 16-bit streams go wide when the tensor
    # is large enough to amortize the extra DSM footprint; fp32 (~12B/elem in
    # flight) stays at 32768 to respect the ~448KB/program DSM budget.
    if N <= 1024:
        return 1024
    if N <= 8192:
        return triton.next_power_of_2(N)
    if N <= 65536:
        return 8192
    if is_fp32:
        return 32768
    if N <= 2**24:
        return 32768
    return 65536


def _fast_path_eligible(x, y, out):
    if not (torch.is_tensor(x) and torch.is_tensor(y)):
        return False
    if x.dtype not in _FAST_DTYPES or y.dtype != x.dtype:
        return False
    if x.shape != y.shape or not (x.is_contiguous() and y.is_contiguous()):
        return False
    if x.numel() == 0 or x.numel() > _MAX_FLAT_NUMEL:
        return False
    if out is not None:
        if not torch.is_tensor(out):
            return False
        if out.dtype != x.dtype or out.shape != x.shape or not out.is_contiguous():
            return False
    return True


def _launch(x, y, out):
    N = x.numel()
    is_fp32 = x.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        logaddexp2_kernel[(grid_size,)](
            x,
            y,
            out,
            N,
            BLOCK=BLOCK,
            EVEN=(N % BLOCK == 0),
            num_warps=2,
        )


def logaddexp2(self, other):
    logger.debug("GEMS_ENFLAME LOGADDEXP2")
    if _fast_path_eligible(self, other, None):
        out = torch.empty_like(self)
        _launch(self, other, out)
        return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS_ENFLAME LOGADDEXP2_OUT")
    if _fast_path_eligible(self, other, out):
        _launch(self, other, out)
        return out
    logaddexp2_func(self, other, out0=out)
    return out
