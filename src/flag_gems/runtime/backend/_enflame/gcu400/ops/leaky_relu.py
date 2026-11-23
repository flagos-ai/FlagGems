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

# Standard flat grid-stride family pattern (same as square/rsqrt/relu): an in-kernel strided loop covers large shapes,
# with the grid capped at 48 CTAs (24 SIP × 2); the EVEN tier (n % BLOCK == 0) needs no mask.
# The BLOCK/num_warps ladder was chosen by interleaved in-process do_bench A/B on this op's body (where + mul, with a
# cvt chain for 16-bit),
# using w1000/r100 medians, on GCU400/L600, 2026-11-16:
#   16-bit big 2^29: 131072/nw4 709.5 (fp16, native 713.8) / 700.7 (bf16,
#     native 702.8) — wide warps hide the cvt chain, confirmed by both out-of-place and in-place (681.6/698.9);
#   16-bit mid 2^24: 131072/nw4 28.4/28.7 (old 8192/nw4 42.1/43.7);
#   fp32 big 2^29: 65536/nw2 1410.3 (native 1411.4); in-place 1352.8 overtakes native;
#   fp32 mid 2^24: 32768/nw2 50.0 (old 8192/nw4 57.5).
# Only one num_warps per (dtype, BLOCK) (the libentry dispatch key does not include nw, to avoid cache collisions).
# For small shapes (≤4096), single-CTA npo2/nw4 and multi-CTA are all on par (a 3.2µs launch floor); keep the old
# configuration.
#
# 2026-11-19 (second pass, dtype-native slope + pipelined strided loop): in-process do_bench medians on GCU400 show
# that `x * negative_slope` with a python-float slope promotes the 16-bit multiply to fp32 (an extra cvt chain per
# element), costing 15-55% of the achievable 1R1W bandwidth. Materialising the slope in the *tensor* dtype with
# `tl.full([1], negative_slope, x.dtype)` keeps the multiply in the native type and matches aten's own semantics:
#   2^26 fp16: 1728 -> 2059 GB/s (native 1959);  2^26 bf16: 1602 -> 2031 GB/s (native 1960);
#   2^29 fp16:  951 -> 2161 GB/s (native 2104);  2^29 bf16: 1697 -> 2145 GB/s.
# The strided loop also wants software pipelining for 16-bit (`num_stages=2` on tl.range): bf16 2^29 goes
# 1723 -> 2145 GB/s only when pipelining and the native-dtype slope are both on. fp32 is unaffected (slope is
# already fp32, so `tl.full` is a no-op) and keeps PIPE=1 to avoid perturbing its already-native-level 2125 GB/s.
NUM_SIPS = 24
GRID_CAP = NUM_SIPS * 2
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# int32 safety domain: host guard keeps numel + BLOCK < 2^31 so that
# block_id * BLOCK + arange (int32) cannot wrap negative on the tail block.
_FLAT_MAX_NUMEL = (1 << 31) - (1 << 17)


@libentry()
@triton.jit(do_not_specialize=["N_total", "negative_slope"])
def _leaky_relu_kernel(
    input_ptr,
    output_ptr,
    N_total,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
    PIPE: tl.constexpr,
    NATIVE_SLOPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=PIPE):
        off = block_id * BLOCK_SIZE + arange
        if EVEN:
            x = tl.load(input_ptr + off)
            if NATIVE_SLOPE:
                # Keep the negative-slope multiply in the tensor dtype (no fp32 promotion).
                slope = tl.full([1], negative_slope, x.dtype)
                tl.store(output_ptr + off, tl.where(x >= 0, x, x * slope))
            else:
                tl.store(output_ptr + off, tl.where(x >= 0, x, x * negative_slope))
        else:
            mask = off < N_total
            x = tl.load(input_ptr + off, mask=mask)
            if NATIVE_SLOPE:
                slope = tl.full([1], negative_slope, x.dtype)
                tl.store(
                    output_ptr + off, tl.where(x >= 0, x, x * slope), mask=mask
                )
            else:
                tl.store(
                    output_ptr + off,
                    tl.where(x >= 0, x, x * negative_slope),
                    mask=mask,
                )


def _next_pow2(numel):
    return 1 << (numel - 1).bit_length() if numel > 1 else 1


def _flat_config(numel, dtype):
    if dtype not in _FLAT_DTYPES:
        # Non-float16/bfloat16/float32 keep the old conservative configuration (same math body, only skipping the EVEN
        # mask)
        if numel <= 1024:
            return 1024, 4
        if numel <= 32768:
            return _next_pow2(numel), 4
        return 8192, 4
    if numel <= 4096:
        block = _next_pow2(numel)
        return (256 if block < 256 else block), 4
    if dtype is torch.float32:
        if numel <= 65536:
            return 2048, 2
        if numel <= (1 << 21):
            return 16384, 2
        if numel <= (1 << 26):
            return 32768, 2
        return 65536, 2
    if numel <= (1 << 21):
        return 8192, 2
    if numel <= (1 << 23):
        return 65536, 2
    return 131072, 4


def _pipe_stages(dtype, numel):
    """Extra pipeline stage on the strided loop.

    16-bit huge tensors (>= 2^27) only reach native bandwidth with `num_stages=2`
    (bf16 2^29: 1723 -> 2145 GB/s), while the 2^24..2^27 mid band is insensitive to
    it in isolation but loses ~40% under memory pressure when pipelined (an extra
    128 KB in-flight tile per stage). Keep PIPE=1 for the mid band and fp32.
    """
    if dtype in (torch.float16, torch.bfloat16) and numel >= (1 << 27):
        return 2
    return 1


def _launch_leaky_relu(input_ptr, output_ptr, n_elements, negative_slope):
    if n_elements == 0:
        return
    block_size, num_warps = _flat_config(n_elements, input_ptr.dtype)
    grid = min(triton.cdiv(n_elements, block_size), GRID_CAP)
    even = n_elements % block_size == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (square/rsqrt precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(input_ptr.device):
                _leaky_relu_kernel[(grid,)](
                    input_ptr,
                    output_ptr,
                    n_elements,
                    negative_slope,
                    BLOCK_SIZE=block_size,
                    EVEN=even,
                    PIPE=_pipe_stages(input_ptr.dtype, n_elements),
                    NATIVE_SLOPE=input_ptr.dtype
                    in (torch.float16, torch.bfloat16),
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "leaky_relu flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _over_int32_domain(n_elements):
    return n_elements >= _FLAT_MAX_NUMEL


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU")
    n = A.numel()
    if _over_int32_domain(n):
        from flag_gems.ops.leaky_relu import leaky_relu as _public

        return _public(A, negative_slope)
    if not A.is_contiguous():
        A = A.contiguous()
    output = torch.empty_like(A)
    _launch_leaky_relu(A, output, n, negative_slope)
    return output


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_")
    n = A.numel()
    if _over_int32_domain(n):
        from flag_gems.ops.leaky_relu import leaky_relu_ as _public

        return _public(A, negative_slope)
    if not A.is_contiguous():
        # A non-contiguous view cannot be addressed linearly by a flat 1D address; first run the same Triton kernel
        # on a contiguous copy, then write back with the original strides, preserving in-place semantics and the
        # returned alias.
        # negative_slope must be passed through, otherwise it silently falls back to the default 0.01.
        A.copy_(leaky_relu(A, negative_slope))
        return A
    _launch_leaky_relu(A, A, n, negative_slope)
    return A


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_OUT")
    n = A.numel()
    if out is None:
        return leaky_relu(A, negative_slope)
    if _over_int32_domain(n):
        from flag_gems.ops.leaky_relu import leaky_relu_out as _public

        return _public(A, negative_slope, out=out)
    if not A.is_contiguous():
        A = A.contiguous()
    _launch_leaky_relu(A, out, n, negative_slope)
    return out
