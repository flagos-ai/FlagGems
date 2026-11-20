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

import builtins
import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.histc import histc as default_histc
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# Upper bound on the number of bins handled by the local-reduce kernel (which
# builds a per-program bin-count vector). Beyond this we use the direct atomic
# kernel that issues one atomic add per element.
MAX_BINS_LOCAL_REDUCE = 1024

NUM_SIPS = 24


@libentry()
@triton.jit
def histc_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    bins: tl.constexpr,
    min_val,
    max_val,
    inv_bin_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Direct histogram kernel.

    Each program loads a contiguous tile of ``BLOCK_SIZE`` elements, computes
    the bin index for every element in fp32 (using int32 indices) and
    atomically increments the matching global bin counter. Elements outside
    ``[min_val, max_val]`` and NaNs are ignored. Used when the bin count is
    too large for the local-reduce kernel.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)

    for block_id in tl.range(
        pid, (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, num_pids
    ):
        offset = block_id * BLOCK_SIZE + arange
        mask = offset < n_elements

        inp_val = tl.load(inp_ptr + offset, mask=mask, other=float("nan")).to(
            tl.float32
        )

        # Elements equal to max go to the last bin; out-of-range/NaN ignored.
        in_range = (inp_val >= min_val) & (inp_val <= max_val)

        # Compute bin index in fp32, then cast to int32 (mthreads has no int64).
        bin_idx = tl.floor((inp_val - min_val) * inv_bin_width).to(tl.int32)
        bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
        # Clamp stray indices (only the in-range ones are stored anyway).
        bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
        bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)

        valid_mask = mask & in_range
        ones = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
        tl.atomic_add(out_ptr + bin_idx, ones, mask=valid_mask, sem="relaxed")


HISTC_LOCAL_REDUCE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
]


def _prune_local_reduce_configs(configs, nargs, **meta):
    n = meta.get("n_elements", nargs["n_elements"])
    out = []
    for cfg in configs:
        bs = cfg.kwargs["BLOCK_SIZE"]
        # A program loops over ceil(n / BLOCK_SIZE) tiles. When BLOCK_SIZE is
        # much larger than n, only one program does useful work and the others
        # sit idle; prefer configs that keep a reasonable number of programs.
        if bs <= n:
            out.append(cfg)
    # Always keep the smallest config as a fallback for tiny inputs.
    if not out and configs:
        out = [configs[0]]
    return out


@libentry()
@triton.autotune(
    configs=HISTC_LOCAL_REDUCE_CONFIGS,
    key=["n_elements", "BINS_PADDED"],
    reset_to_zero=["out_ptr"],
    prune_configs_by={"early_config_prune": _prune_local_reduce_configs},
)
@triton.jit
def histc_local_reduce_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    min_val,
    max_val,
    inv_bin_width,
    BLOCK_SIZE: tl.constexpr,
    BINS: tl.constexpr,
    BINS_PADDED: tl.constexpr,
):
    """Histogram kernel with a per-program local reduce.

    Each program loads a tile, builds a local bin-count vector of length
    ``BINS_PADDED`` (a power-of-two >= ``BINS``) with the hardware-accelerated
    ``tl.histogram`` intrinsic, and only issues ``BINS`` (or fewer) atomic adds
    to the global histogram instead of ``BLOCK_SIZE``. This greatly reduces
    global-memory atomic contention when the bin count is modest relative to
    the tile size.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    bin_arange = tl.arange(0, BINS_PADDED)
    bin_mask = bin_arange < BINS

    for block_id in tl.range(
        pid, (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, num_pids
    ):
        offset = block_id * BLOCK_SIZE + arange
        mask = offset < n_elements

        inp_val = tl.load(inp_ptr + offset, mask=mask, other=float("nan")).to(
            tl.float32
        )

        in_range = (inp_val >= min_val) & (inp_val <= max_val)
        bin_idx = tl.floor((inp_val - min_val) * inv_bin_width).to(tl.int32)
        bin_idx = tl.where(inp_val == max_val, BINS - 1, bin_idx)
        # Invalid lanes are masked out of the histogram; clamp to keep indices
        # in range so the intrinsic never sees out-of-bounds values.
        bin_idx = tl.where(in_range, bin_idx, 0)

        valid_mask = mask & in_range

        # Local histogram over the tile (hardware intrinsic).
        local_counts = tl.histogram(bin_idx, BINS_PADDED, mask=valid_mask).to(
            tl.float32
        )

        # Flush the non-zero local counts to the global histogram.
        flush_mask = bin_mask & (local_counts > 0)
        tl.atomic_add(
            out_ptr + bin_arange, local_counts, mask=flush_mask, sem="relaxed"
        )


def _use_triton_kernel(inp: torch.Tensor, bins: int) -> Tuple[bool, int]:
    if not isinstance(inp, torch.Tensor):
        return False, 0
    if inp.device.type != "musa" or inp.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if not inp.is_contiguous() or inp.numel() == 0:
        return False, 0
    if bins <= 0:
        return False, 0
    return True, inp.element_size()


def _resolve_range(inp: torch.Tensor, min_val: float, max_val: float):
    """Resolve the histogram range, mirroring torch.histc semantics."""
    if min_val == 0 and max_val == 0:
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())
    return min_val, max_val


def histc(inp, bins=100, min=0, max=0):
    """Compute the histogram of a tensor (mthreads specialized)."""
    logger.debug("GEMS_MTHREADS HISTC")

    use_triton, _ = _use_triton_kernel(inp, bins)
    if not use_triton:
        return default_histc(inp, bins=bins, min=min, max=max)

    inp = inp.contiguous()

    min_val = float(min)
    max_val = float(max)
    min_val, max_val = _resolve_range(inp, min_val, max_val)

    out_dtype = inp.dtype
    # Accumulate in fp32: atomic_add on fp16/bf16 is unsupported on mthreads,
    # and fp32 accumulation avoids the per-element atomic on the small output.
    acc = torch.zeros(bins, dtype=torch.float32, device=inp.device)
    n_elements = inp.numel()

    # Degenerate range: all in-range elements equal min_val, counted in bin 0.
    if min_val == max_val:
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        acc[0] = count
        return acc.to(out_dtype)

    inv_bin_width = float(bins) / (max_val - min_val)

    # Use the local-reduce kernel when the bin count is small enough: it
    # amortizes the global atomics across the whole tile and is markedly faster
    # for moderate bin counts. Fall back to the direct atomic kernel otherwise.
    if bins <= MAX_BINS_LOCAL_REDUCE:
        bins_padded = triton.next_power_of_2(bins)
        # Cap the grid to a moderate number of programs; each program streams
        # over multiple tiles, so a small grid is enough to saturate the device
        # while keeping the autotune key stable.
        num_programs = builtins.min(triton.cdiv(n_elements, 1024), NUM_SIPS * 4)
        grid = lambda meta: (
            builtins.min(
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                num_programs,
            ),
        )
        with torch_device_fn.device(inp.device):
            histc_local_reduce_kernel[grid](
                inp,
                acc,
                n_elements,
                min_val,
                max_val,
                inv_bin_width,
                BINS=bins,
                BINS_PADDED=bins_padded,
            )
    else:
        block_size = 8192
        n_blocks = (n_elements + block_size - 1) // block_size
        grid = builtins.min(n_blocks, NUM_SIPS * 2)
        with torch_device_fn.device(inp.device):
            histc_kernel[(grid,)](
                inp,
                acc,
                n_elements,
                bins,
                min_val,
                max_val,
                inv_bin_width,
                BLOCK_SIZE=block_size,
                num_warps=4,
                num_stages=2,
            )

    return acc.to(out_dtype)
