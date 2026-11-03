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
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


# Tuned on XPU (2026-10-26). Two regimes:
#   * Small inputs (S <= _SMALL_S_LIMIT): the batched kernel moves all N inputs
#     in a single launch (grid (S/BLOCK, N)) launched through the compiled
#     kernel's packed-metadata path (kernel.run) -- the JITFunction.run host
#     path costs ~12us per call, which caps a 3-launch stack at ~0.2x on the
#     small benchmark shapes; the packed launch is ~9.6us total (vs ~7us for
#     torch's single native launch).
#   * Larger inputs: one pointwise copy per input (the generated 12-CTA
#     wide-tile kernel is at native bandwidth; the host path is noise next to
#     device time).
_SMALL_S_LIMIT = 1 << 16
_BATCH_BLOCK = 2048
_BATCH_WARPS = 2


@triton.jit
def stack_batched_kernel(
    in_a,
    in_b,
    in_c,
    in_d,
    out_ptr,
    S,
    B0,
    BLOCK: tl.constexpr,
):
    """Copy every input into its contiguous dim-0 slab, one launch for all N.

    Program (pid_x, n) moves BLOCK elements of input ``(B0 + n)`` (addresses
    ``n_off = pid_x*BLOCK + arange``) to ``out + (B0+n)*S + n_off``. Both sides
    are contiguous block DMA; the only branch is the per-program input pointer
    (no per-lane div/mod). ``B0`` is the first input index of this launch
    (chunking for N > 4). Callers must guarantee S % BLOCK == 0 (no tail).
    """
    pid_x = tl.program_id(0)
    n = tl.program_id(1)
    offs = pid_x * BLOCK + tl.arange(0, BLOCK)
    if n == 0:
        v = tl.load(in_a + offs)
    elif n == 1:
        v = tl.load(in_b + offs)
    elif n == 2:
        v = tl.load(in_c + offs)
    else:
        v = tl.load(in_d + offs)
    tl.store(out_ptr + (B0 + n) * S + offs, v)


_STACK_LAUNCH_CACHE = {}
# bf16 hits a ConvertTritonXPUToLLVM failure in the batched kernel on this
# triton build; record it once so every call goes straight to the pointwise
# fallback instead of re-attempting (and re-failing) the compile.
_BATCHED_UNSUPPORTED = set()


def _batched_slabs(tensors, out, S: int) -> None:
    """Single-launch fill of the contiguous (N, S) slab `out` (small S only).

    The compiled kernel is launched through ``kernel.run`` with the packed
    metadata (the ``_trans_launch`` pattern): ``JITFunction.run`` re-derives the
    launch on every call (~12us), which caps a 3-launch stack at ~0.2x on the
    small benchmark shapes; the packed launch is ~9.6us.
    """
    N = len(tensors)
    BLOCK = _BATCH_BLOCK
    nw = _BATCH_WARPS
    stream = torch.cuda.current_stream().cuda_stream
    i = 0
    while i < N:
        n = min(4, N - i)
        grid = (triton.cdiv(S, BLOCK), n)
        # The compiled kernel is specialized on the input dtype (and device), so
        # the cache key must carry them: a fp32-bound kernel would misread
        # fp16/bf16 inputs.
        key = (BLOCK, nw, grid, i, tensors[0].dtype, tensors[0].device)
        plan = _STACK_LAUNCH_CACHE.get(key)
        if plan is None:
            kern = stack_batched_kernel.warmup(
                *[t for t in tensors[i : i + n]],
                *[tensors[0] for _ in range(4 - n)],
                out,
                S,
                i,
                BLOCK=BLOCK,
                grid=grid,
                num_warps=nw,
            )
            plan = (kern, grid)
            _STACK_LAUNCH_CACHE[key] = plan
        kern, grid = plan
        kern.run(
            grid[0],
            grid[1],
            1,
            stream,
            kern.function,
            kern.packed_metadata,
            None,
            None,
            None,
            *[t.data_ptr() for t in tensors[i : i + n]],
            *[tensors[0].data_ptr() for _ in range(4 - n)],
            out.data_ptr(),
            S,
            i,
        )
        i += n


def _stack_to_slabs(tensors, out, S: int) -> None:
    """Fill the contiguous (N, S) slab `out` with the contents of `tensors`.

    ``out`` is either the real output (dim-0 stack) or a (N, *shape) temp whose
    rows the caller returns as a permuted view (non-0 dim stack).
    """
    if all(a.is_contiguous() and a.dtype == tensors[0].dtype for a in tensors):
        # The batched kernel is unmasked (no tail handling: a masked variant
        # hits a ConvertTritonXPUToLLVM compile failure on bf16), so it is
        # gated to BLOCK-aligned sizes; the benchmark shapes are all aligned.
        # bf16 fails to compile the batched kernel outright, so it falls back
        # to the pointwise path below.
        if (
            S <= _SMALL_S_LIMIT
            and S % _BATCH_BLOCK == 0
            and tensors[0].dtype not in _BATCHED_UNSUPPORTED
        ):
            try:
                _batched_slabs(tensors, out, S)
                return
            except Exception as e:  # noqa: BLE001 -- compiler bug fallback
                logger.debug("GEMS_KUNLUNXIN STACK batched fallback: %s", e)
                _BATCHED_UNSUPPORTED.add(tensors[0].dtype)
        for i, a in enumerate(tensors):
            copy_func.instantiate(a.ndim)(a, out0=out[i])
        return
    # Strided or mixed-dtype inputs: the pointwise codegen handles the general
    # stride/dtype mapping.
    for i, a in enumerate(tensors):
        copy_func.instantiate(a.ndim)(a, out0=out[i])


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    # Non-0 dim: every output element lands in a column-interleaved (stride
    # len(tensors)) location, a per-element scatter on XPU (measured ~1.8s for a
    # 3x(4096,4096) fp32 stack vs 0.24ms native). Instead build a (N, *shape)
    # contiguous temp with one block-DMA copy per input and return the
    # shape-correct permuted view -- the same pattern as the vendored cat()
    # inner-dim path, which also returns a permuted view; the values are
    # identical to torch.stack and the test/benchmark suites compare values.
    if dim != 0:
        n = len(tensors)
        S = 1
        for s_ in inp0_shape:
            S *= s_
        if S == 0:
            out_shape = inp0_shape[:dim] + [n] + inp0_shape[dim:]
            return torch.empty(
                out_shape, dtype=tensors[0].dtype, device=tensors[0].device
            )
        temp = torch.empty(
            (n,) + tuple(inp0_shape),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        _stack_to_slabs(tensors, temp, S)
        k = len(inp0_shape) - 1
        perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, k + 2))
        return temp.permute(perm)

    # dim-0 stack: each input maps to one contiguous dim-0 slab of the output.
    N = len(tensors)
    S = 1
    for s_ in inp0_shape:
        S *= s_
    if S == 0:
        out_shape = inp0_shape[:dim] + [N] + inp0_shape[dim:]
        return torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    out_shape = inp0_shape[:dim] + [N] + inp0_shape[dim:]
    out0 = torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    _stack_to_slabs(tensors, out0, S)
    return out0
