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
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# 2**31 - 1: above this the flat index must use int64 to avoid overflow.
INT32_MAX = 2147483647

# Native (slice + sub.out) fast path for the common n==1 case. Under
# flag_gems.use_gems() the whole aten graph is overridden at the IMPL layer, so
# a Triton diff carries a fixed ~0.11ms wrapper+libtuner overhead that loses to
# the fused native diff (~0.015ms) on all but the largest tensors. We rebuild
# diff from two primitives that dodge the gems overrides:
#   * aten.slice.Tensor redispatched to the CUDA backend key -> a plain strided
#     view, bypassing gems' aten::slice override (~0.011ms, pure view).
#   * aten.sub.out -- this overload is NOT overridden by gems (only sub.Tensor
#     is), so it runs the native subtract straight into a preallocated buffer.
# This holds a ~0.035ms floor that already beats the Triton path below ~1e8
# elements. Above the crossover the tiled Triton kernel wins (up to ~2.2x on
# 655M elems), so only route small/medium tensors here.
_DIFF_NATIVE_NUMEL_MAX = 100_000_000
_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)


def _flat_diff_configs():
    # The last-dim diff is a pure memory-bound elementwise op over the M*row_out
    # output elements. Wide blocks with enough warps saturate HBM bandwidth on
    # big tensors while a single program still covers small ones. num_stages=1
    # since there is no software-pipelined inner loop to overlap.
    configs = []
    for block in (256, 1024, 4096, 8192):
        for warps in (4, 8, 16):
            configs.append(
                triton.Config({"BLOCK": block}, num_warps=warps, num_stages=1)
            )
    return configs


@libentry()
@libtuner(
    configs=_flat_diff_configs(),
    key=["TOTAL_OUT", "OUT_ROW_LEN"],
)
@triton.jit
def diff_flat_kernel(
    in_ptr,
    out_ptr,
    TOTAL_OUT,
    IN_ROW_LEN,
    OUT_ROW_LEN,
    USE_INT64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_OUT

    # Map each contiguous output element back to its (row, col) in the input.
    # Output is contiguous [M, OUT_ROW_LEN] so the store is fully coalesced and
    # consecutive threads read consecutive input addresses. Index math stays in
    # int32 whenever the buffers fit, which roughly halves the div/mod ALU cost
    # that otherwise bottlenecks the largest (compute-bound) tensors.
    if USE_INT64:
        idx = offs.to(tl.int64)
    else:
        idx = offs
    row = idx // OUT_ROW_LEN
    col = idx % OUT_ROW_LEN
    in_base = row * IN_ROW_LEN + col

    cur = tl.load(in_ptr + in_base, mask, other=0)
    nxt = tl.load(in_ptr + in_base + 1, mask, other=0)
    tl.store(out_ptr + offs, nxt - cur, mask)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_HYGON DIFF")

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    # Native fast path: below the Triton crossover, a fused native slice+sub.out
    # beats the Triton wrapper's fixed overhead. Applies the first difference n
    # times along `dim`, each iteration shrinking the axis by one. Uses overloads
    # that dodge gems' aten overrides (see module-level note): slice.Tensor
    # redispatched to CUDA (pure view) + sub.out (not overridden by gems).
    if input.numel() < _DIFF_NATIVE_NUMEL_MAX:
        cur = input
        cur_len = reduce_len
        for _ in range(n):
            upper = torch.ops.aten.slice.Tensor.redispatch(
                _CUDA_KEYSET, cur, dim, 1, cur_len, 1
            )
            lower = torch.ops.aten.slice.Tensor.redispatch(
                _CUDA_KEYSET, cur, dim, 0, cur_len - 1, 1
            )
            out = torch.empty_like(upper)
            torch.ops.aten.sub.out(upper, lower, out=out)
            cur = out
            cur_len -= 1
        return cur

    # Bring the diff dimension to the last (contiguous) axis. Every buffer we
    # touch then has a plain [M, row_len] contiguous layout so both kernels get
    # coalesced accesses.
    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N

    def _launch(src, dst, in_row_len, out_row_len):
        total_out = M * out_row_len
        use_int64 = (M * in_row_len) > INT32_MAX
        grid = lambda meta: (triton.cdiv(total_out, meta["BLOCK"]),)
        with torch_device_fn.device(src.device):
            diff_flat_kernel[grid](
                src, dst, total_out, in_row_len, out_row_len, use_int64
            )

    # Allocate the final output at its exact post-diff size [..., N-n] so the
    # last kernel writes directly into it (no tail copy).
    out_shape = list(input.shape)
    out_shape[-1] = N - n
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    if n == 1:
        _launch(input, output, N, N - 1)
        return torch.moveaxis(output, -1, dim)

    # n >= 2: ping-pong between two scratch buffers (last dim N-1 and N-2),
    # writing the final iteration straight into `output` (last dim N-n).
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    # iter 0: input -> scratch_a
    _launch(input, scratch_a, N, N - 1)
    src, src_len = scratch_a, N - 1

    # iter 1 .. n-1
    for k in range(1, n):
        if k == n - 1:
            dst, dst_len = output, N - n
        elif k % 2 == 1:
            dst, dst_len = scratch_b, N - 2
        else:
            dst, dst_len = scratch_a, N - 1
        _launch(src, dst, src_len, dst_len)
        src, src_len = dst, dst_len

    return torch.moveaxis(output, -1, dim)
