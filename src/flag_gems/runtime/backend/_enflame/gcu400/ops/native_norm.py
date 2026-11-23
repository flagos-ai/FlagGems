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

from flag_gems.ops.vector_norm import vector_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 exposes 48 SPCs; one program per SPC keeps every lane of the
# accumulator resident without cross-SPC contention.
_MAX_GRID = 48


@libentry()
@triton.jit
def _native_norm_l2_kernel(X, Out, M, BLOCK_SIZE: tl.constexpr):
    """Single-block L2 norm of a contiguous full reduction.

    `native_norm` always reduces the whole tensor, so the partial-then-final
    two-kernel scheme in `vector_norm` is pure overhead here: for small inputs
    the second launch costs as much as the reduction itself. One block covers
    the whole tensor below the cutoff used by the caller.
    """
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    x = tl.load(X + offset, mask=mask, other=0.0).to(acc_dtype)
    tl.store(Out, tl.sqrt(tl.sum(x * x)))


@libentry()
@triton.jit(do_not_specialize=["M"])
def _native_norm_l2_partial(
    X, Mid, M, BLOCK: tl.constexpr, num_stages: tl.constexpr = 1
):
    """Persistent partial L2 reduction for medium/large tensors.

    Each program owns a strided set of full blocks and folds them into a
    register accumulator with pure element-wise FFMAs — no cross-lane
    traffic inside the loop. Only one BLOCK-lane reduction tree per program
    runs at the very end, so the tree cost is amortised over the whole
    tensor instead of being paid once per tile.
    """
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_full = M // BLOCK

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for chunk in tl.range(pid, num_full, num_prog, num_stages=num_stages):
        offset = chunk * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset).to(tl.float32)
        acc += x * x

    # The single masked tail block is handled by the program that owns it in
    # the strided schedule, keeping the hot loop free of any mask predicate.
    if pid == num_full % num_prog:
        offset = num_full * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset, mask=offset < M, other=0.0).to(tl.float32)
        acc += x * x

    tl.store(Mid + pid, tl.sum(acc))


@libentry()
@triton.jit
def _native_norm_l2_single(X, Mid, BLOCK: tl.constexpr):
    """One program per full block: no loop, no tail, no mask predicate.

    For inputs that divide into a handful of 32KB blocks the persistent
    strided loop degenerates to one iteration per program, so the loop
    control and the tail-block predicate are pure overhead; a flat one
    block-per-program kernel with 32KB chunks runs measurably faster
    (64K elements: 4.73us vs 5.36us, three dtypes, 5-round medians).
    """
    pid = tl.program_id(0)
    x = tl.load(X + pid * BLOCK + tl.arange(0, BLOCK)).to(tl.float32)
    tl.store(Mid + pid, tl.sum(x * x))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def _native_norm_l2_final(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid = tl.load(Mid + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out, tl.sqrt(tl.sum(mid)))


def _partial_block_size(numel, cap):
    # Keep every SPC busy on small tensors (narrow chunks -> more programs)
    # and widen the per-program load granularity up to the cap on large ones.
    # fp32 caps at 32K (widest load beats the extra tree depth), fp16/bf16 at
    # 16K; both keep the accumulator + load buffers far below the DSM budget.
    target = triton.next_power_of_2((numel + _MAX_GRID - 1) // _MAX_GRID)
    return min(cap, max(2048, target))


# Largest element count handled by the fused single-block kernel. Below this
# the fused kernel beats the two-kernel scheme (the saved launch dominates);
# above it the two-kernel scheme wins because one block can no longer hide
# the load latency. The value also keeps BLOCK_SIZE within triton's maximum
# tensor numel.
_FUSED_MAX_NUMEL = 16384

# Sizes that divide into whole 32KB blocks go one-block-per-program instead
# of the persistent strided loop: fewer, wider chunks beat more, narrower
# programs once the tensor spans only a handful of DMA transfers. Above this
# many blocks the loop amortises and the wider fp32 chunks below take over.
_SINGLE_BLOCK_NUMEL = 8192
_SINGLE_MAX_NUMEL = 256 * 1024


# Crossover measured on GCU400 (do_bench median, interleaved A/B): below it,
# dense 32KB chunks keep all 48 SPCs streaming (1M: 6.96us vs 8.34; 8M: 23.4
# vs 24.0); above it, 128KB chunks on 24 programs already saturate the
# aggregate DRAM ceiling with less per-program ramp-up waste (64M: 127.8 vs
# 136.7). The two schedules tie exactly at the crossover (12M: 31.4 both).
_FP32_WIDE_MIN_NUMEL = 12 * 1024 * 1024
_FP32_SMALL_MAX_NUMEL = 256 * 1024


def _l2_two_stage(x, out, numel):
    if (
        _FUSED_MAX_NUMEL < numel <= _SINGLE_MAX_NUMEL
        and numel % _SINGLE_BLOCK_NUMEL == 0
    ):
        # Whole 32KB blocks, at most 32 programs: flat one-block-per-program.
        single = True
        block, grid_size, num_warps = (
            _SINGLE_BLOCK_NUMEL,
            numel // _SINGLE_BLOCK_NUMEL,
            4,
        )
    else:
        single = False
        if x.dtype == torch.float32:
            if numel >= _FP32_WIDE_MIN_NUMEL:
                block, grid_size, num_warps = 32768, 24, 4
            elif numel <= _FP32_SMALL_MAX_NUMEL:
                block, grid_size, num_warps = (
                    _partial_block_size(numel, 8192),
                    _MAX_GRID,
                    4,
                )
            else:
                block, grid_size, num_warps = 8192, 48, 4
        else:
            # 16-bit lanes move half the bytes per element, capping
            # per-program throughput; the 48-way adaptive schedule measured
            # fastest at every size probed, so 16-bit inputs keep it.
            block, grid_size, num_warps = (
                _partial_block_size(numel, 16384),
                _MAX_GRID,
                4,
            )
        num_chunks = triton.cdiv(numel, block)
        grid_size = min(num_chunks, grid_size)
        num_stages = 2 if num_chunks > grid_size else 1

    mid = torch.empty([grid_size], dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        if single:
            _native_norm_l2_single[(grid_size,)](
                x, mid, BLOCK=block, num_warps=num_warps
            )
        else:
            _native_norm_l2_partial[(grid_size,)](
                x, mid, numel, BLOCK=block, num_stages=num_stages, num_warps=num_warps
            )
        _native_norm_l2_final[(1,)](
            mid,
            out,
            grid_size,
            BLOCK_MID=triton.next_power_of_2(grid_size),
            num_warps=1,
        )


def native_norm(self, p=2):
    """
    Compute the p-norm over all elements of the input tensor.

    Args:
        self: Input tensor
        p: The order of norm (default: 2)

    Returns:
        A scalar tensor containing the p-norm
    """
    logger.debug("GEMS_ENFLAME NATIVE_NORM")

    # Validate dtype
    if self.dtype not in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
        raise NotImplementedError(f"native_norm not implemented for {self.dtype}")

    numel = self.numel()
    if float(p) == 2.0 and self.dtype != torch.float64:
        if numel == 0:
            out = torch.empty([], dtype=self.dtype, device=self.device)
            with torch_device_fn.device(out.device):
                out.fill_(0.0)
            return out

        # Small inputs: one launch, one block. Medium/large inputs: the
        # persistent two-stage scheme above (the generic vector_norm pays
        # one full reduction tree per sqrt(M)-sized tile, which is an
        # order of magnitude slower on this backend).
        if numel <= _FUSED_MAX_NUMEL:
            x = self.contiguous()
            out = torch.empty([], dtype=self.dtype, device=x.device)
            with torch_device_fn.device(x.device):
                _native_norm_l2_kernel[(1,)](
                    x, out, x.numel(), BLOCK_SIZE=triton.next_power_of_2(x.numel())
                )
            return out

        x = self if self.is_contiguous() else self.contiguous()
        out = torch.empty([], dtype=self.dtype, device=x.device)
        _l2_two_stage(x, out, numel)
        return out

    # Delegate to vector_norm with dim=None (full reduction), keepdim=False.
    return vector_norm(self, ord=p, dim=None, keepdim=False, dtype=self.dtype)
