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

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# GCU400 flat fast path for contiguous fp16/bf16/f32 softshrink.
# Tuned on the official core bench shapes (kernel-mode do_bench median,
# w1000/r100, 2026-11-16, GCU400/L600; in-process round-robin probe with a
# copy-floor arm, probe JSONs in the job tmp dir):
#   * The seed where-chain body (f32 cmp/select x3 + NaN select, 8-9 vector
#     ops/elem) is ALU-bound, not memory-bound: 16-bit 512Mi ran at 1.26TB/s
#     vs the ~2.9-3.0TB/s 1R1W flat-stream DRAM roofline (bitwise_not/rsqrt
#     family), i.e. 2.4x off the physical floor.
#   * The clamp identity  softshrink(x) == x - clamp(x, -l, l)  (two
#     minnum/maxnum + one sub; NaN propagates naturally through the final
#     subtraction since clamp(NaN) = l) restores roofline behaviour:
#     16-bit 512Mi 1635 -> 679us, 16-bit 16.7M 54.5 -> 27.2us,
#     fp32 512Mi 1711 -> 1349us, fp32 16.7M 59.4 -> 48.0us.
#     It matches the generic where-chain bit-exactly except the -0/+0 sign
#     corner where both outputs are value-equal (x - clamp(x) = +0 there,
#     same as the generic literal 0.0 arm).
#   * 16-bit dtypes additionally skip the f32 round-trip (NATIVE arm) when
#     lambd is exactly representable in the storage dtype (0.5/1.0/0.0 and
#     friends): fp16 arithmetic on exactly-representable operands rounds the
#     exact difference once, identical to the f32-exact-then-round path.
#     Non-representable lambd falls back to the f32-domain body at the same
#     block config (28.9 vs 27.1us @16.7M fp16).
#   * BLOCK ladder: <=2^16 elems use next_pow2(n//2) (>=512) with num_warps=2
#     to amortize launch latency over several CTAs; larger sizes use 65536
#     with num_warps=2 (16-bit, keeps DSM at 256KB = true 48-way) or
#     num_warps=4 (fp32, bitwise_not int32 precedent). EVEN elides the
#     boundary mask when n % BLOCK == 0 (probe: -0.5% giant / -2.3% fp32 mid;
#     masked arm kept for ragged n). Each (dtype-class, BLOCK) pair maps to
#     exactly one num_warps: LibEntry's dispatch key omits num_warps (relu
#     precedent).
#   * Launch retries: the external llvm converter intermittently fails on
#     cold compiles (bitwise_xor / rad2deg precedents); retrying recompiles.

_SMALL_MAX = 1 << 16
_HUGE_BLOCK = 65536


@libentry()
@triton.jit(do_not_specialize=["N_total", "lambd"])
def softshrink_kernel(
    x_ptr,
    out_ptr,
    N_total,
    lambd,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
    NATIVE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            if NATIVE:
                lam = lambd.to(x.dtype)
                c = tl.minimum(tl.maximum(x, -lam), lam)
                res = x - c
            else:
                x32 = x.to(tl.float32)
                c = tl.minimum(tl.maximum(x32, -lambd), lambd)
                res = (x32 - c).to(x.dtype)
            tl.store(out_ptr + off, res)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask, other=0)
            if NATIVE:
                lam = lambd.to(x.dtype)
                c = tl.minimum(tl.maximum(x, -lam), lam)
                res = x - c
            else:
                x32 = x.to(tl.float32)
                c = tl.minimum(tl.maximum(x32, -lambd), lambd)
                res = (x32 - c).to(x.dtype)
            tl.store(out_ptr + off, res, mask=mask)


def _check_supported_dtype(t: torch.Tensor):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {t.dtype}. Supported dtypes are float16, bfloat16, and float32."
        )


def _lambda_representable(dtype: torch.dtype, lambd: float) -> bool:
    try:
        t = torch.tensor(lambd, dtype=dtype)
    except Exception:
        return False
    return float(t) == float(lambd)


def _choose_config(x: torch.Tensor, n: int, lambd: float):
    itemsize = x.element_size()
    if n <= _SMALL_MAX:
        block = max(512, triton.next_power_of_2(max(n // 2, 1)))
        num_warps = 2
    else:
        block = _HUGE_BLOCK
        num_warps = 2 if itemsize == 2 else 4
    native = (
        itemsize == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and _lambda_representable(x.dtype, lambd)
    )
    return block, num_warps, native


def _launch_softshrink_kernel(x: torch.Tensor, out: torch.Tensor, lambd: float):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block, num_warps, native = _choose_config(x, n_elements, lambd)
    grid = min(triton.cdiv(n_elements, block), NUM_SIPS * 2)
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                softshrink_kernel[(grid,)](
                    x,
                    out,
                    n_elements,
                    float(lambd),
                    BLOCK_SIZE=block,
                    EVEN=(n_elements % block == 0),
                    NATIVE=native,
                    num_warps=num_warps,
                )
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning("softshrink kernel launch failed (attempt %d)", attempt + 1)


def softshrink(input: torch.Tensor, lambd: float = 0.5):
    logger.debug("GEMS_ENFLAME SOFTSHRINK")
    _check_supported_dtype(input)
    x = input.contiguous()
    out = torch.empty_like(x)
    _launch_softshrink_kernel(x, out, lambd)
    return out.reshape_as(input)


def softshrink_out(input: torch.Tensor, lambd: float = 0.5, out: torch.Tensor = None):
    logger.debug("GEMS_ENFLAME SOFTSHRINK_OUT")
    if out is None:
        raise ValueError("Argument 'out' must be provided for softshrink_out.")
    if input.shape != out.shape:
        raise ValueError(
            f"Shape mismatch: input.shape={input.shape}, out.shape={out.shape}"
        )
    if input.dtype != out.dtype:
        raise TypeError(
            f"Dtype mismatch: input.dtype={input.dtype}, out.dtype={out.dtype}"
        )
    _check_supported_dtype(input)

    x = input.contiguous()
    if out.is_contiguous():
        out_buf = out
    else:
        out_buf = torch.empty_like(out, memory_format=torch.contiguous_format)

    _launch_softshrink_kernel(x, out_buf, lambd)

    if out_buf.data_ptr() != out.data_ptr():
        out.copy_(out_buf)
    return out
