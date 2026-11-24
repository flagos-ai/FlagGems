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
import threading

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Output rows handled by one program. ROWS=8 is the measured optimum (an 8x4096
# tile still fits the register budget); anything larger regresses.
_EMBEDDING_ROWS = 8
# Column limit of the single-row tile; for larger N the kernel loops over
# columns.
_EMBEDDING_BLOCK_MAX = 4096


@triton.jit
def _embedding_gather_kernel(
    out_ptr,
    index_ptr,
    weight_ptr,
    M,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IDX_I64: tl.constexpr,
):
    """Copy whole rows of ``weight`` into the output following ``indices``.

    Two bad paths must be avoided on gcu400: reading int64 indices on the
    device returns wrong values (the whole gather is misaligned), and a scalar
    (0-d) index in pointer arithmetic aborts with SIGABRT. The indices are
    therefore normalized to int32 on the host, the kernel keeps a ``[ROWS]``
    vector index, and all pointer arithmetic stays 2D.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M
    if IDX_I64:
        # int64 indices are read as their low int32 word instead of running a separate
        # cast kernel on the host: any embedding index fits in int32, so the low word is
        # the value, and this removes one kernel launch (~4.5 us) from every call, which
        # dominates the latency of the small shapes in this op's benchmark.
        src_rows = tl.load(index_ptr + rows * 2, mask=row_mask, other=0)
    else:
        src_rows = tl.load(index_ptr + rows, mask=row_mask, other=0)
    for col_start in range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        value = tl.load(
            weight_ptr + src_rows[:, None] * N + cols[None, :], mask=mask, other=0.0
        )
        tl.store(out_ptr + rows[:, None] * N + cols[None, :], value, mask=mask)


@libentry()
@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,  # pointer to the input
    elem_cnt: tl.constexpr,  # number of columns in X
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE
    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt
    idx = tl.load(indices + offsets, mask=mask, other=0)
    if idx.dtype != tl.int32:
        idx = idx.to(tl.int32)
    tl.atomic_add(indices_freq + idx, 1, mask=mask)


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    """Embedding forward: gather rows from ``weight`` following ``indices``.

    The original implementation used a scalar pointer offset
    (``weight_ptr += row_idx * N``) while also reading int64 indices on the
    device, which on gcu400 cause SIGABRT and whole-row misalignment
    respectively, so ``tests/test_embedding.py`` failed as a group. This
    version uses host-side int32 indices plus a ``[ROWS, BLOCK_N]`` 2D tile
    and matches the reference bit-exactly in measurement.
    """
    logger.debug("GEMS_ENFLAME EMBEDDING_FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]
    # TODO: remove contiguous enforcement
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    gather_index = indices.reshape(-1)
    # int64 indices are passed through as an int32 view (low word per element) instead of
    # being cast on the device: the cast costs a full extra kernel launch, and reading
    # int64 lanes directly is unreliable on gcu400.
    index_is_i64 = gather_index.dtype == torch.int64
    if index_is_i64:
        gather_index = gather_index.view(torch.int32)
    elif gather_index.dtype != torch.int32:
        gather_index = gather_index.to(torch.int32)

    block_n = min(triton.next_power_of_2(N), _EMBEDDING_BLOCK_MAX)
    grid = (triton.cdiv(M, _EMBEDDING_ROWS),)
    with torch_device_fn.device(weight.device):
        _embedding_gather_kernel[grid](
            output, gather_index, weight, M, N, _EMBEDDING_ROWS, block_n, index_is_i64
        )

    return output


# ---------------------------------------------------------------------------
# Backward: Triton atomic_add on gcu400 measures ~80x slower than aten
# (4096x4096 fp32: 35ms vs 0.43ms, independent of tile shape), so the main path
# forwards to the native aten implementation. Only when flag_gems has already
# taken over aten::embedding_backward (enable scenario) and aten can no longer
# be entered does it fall back to the atomic version below (slower, correct).
# ---------------------------------------------------------------------------
_EMBEDDING_BACKWARD_ROWS = 4
_EMBEDDING_FREQ_BLOCK = 256

# Whether this thread is already forwarding to aten, used to detect reentry in
# the enable scenario.
_NATIVE_BACKWARD_STATE = threading.local()


@triton.jit
def _embedding_backward_atomic_kernel(
    grad_in_ptr,
    grad_out_ptr,
    index_ptr,
    M,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PADDING_IDX: tl.constexpr,
):
    """Fallback path: atomically accumulate each sample's grad rows into the
    target rows.

    The indices must likewise be normalized to int32 on the host (device-side
    int64 reads are unreliable) and keep a ``[ROWS]`` vector index with 2D
    pointer arithmetic.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M
    dst_rows = tl.load(index_ptr + rows, mask=row_mask, other=0)
    for col_start in range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        if PADDING_IDX >= 0:
            mask = mask & (dst_rows[:, None] != PADDING_IDX)
        value = tl.load(
            grad_out_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0
        )
        tl.atomic_add(
            grad_in_ptr + dst_rows[:, None] * N + cols[None, :], value, mask=mask
        )


def _embedding_backward_atomic(
    grad_outputs, indices, num_weights, padding_idx, scale_grad_by_freq
):
    """Backward implementation that does not rely on aten (correct, but clearly
    slower than native on gcu400)."""
    M = indices.numel()
    N = grad_outputs.shape[-1]
    dtype = grad_outputs.dtype
    # half/bfloat16 accumulate in fp32, matching the aten reference behavior.
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    grad_inputs = torch.zeros(
        (num_weights, N), device=grad_outputs.device, dtype=acc_dtype
    )
    grad_out = grad_outputs.reshape(M, N)
    if acc_dtype != dtype:
        grad_out = grad_out.to(acc_dtype)

    gather_index = indices.reshape(-1)
    if gather_index.dtype != torch.int32:
        gather_index = gather_index.to(torch.int32)

    block_n = min(triton.next_power_of_2(N), _EMBEDDING_BLOCK_MAX)
    grid = (triton.cdiv(M, _EMBEDDING_BACKWARD_ROWS),)
    normalized_padding_idx = -1 if padding_idx is None else int(padding_idx)

    with torch_device_fn.device(grad_outputs.device):
        _embedding_backward_atomic_kernel[grid](
            grad_inputs,
            grad_out,
            gather_index,
            M,
            N,
            _EMBEDDING_BACKWARD_ROWS,
            block_n,
            normalized_padding_idx,
        )
        if scale_grad_by_freq:
            freq = torch.zeros(
                (num_weights,), device=grad_outputs.device, dtype=torch.int32
            )
            indice_freq_kernel[(triton.cdiv(M, _EMBEDDING_FREQ_BLOCK),)](
                freq, gather_index, M, _EMBEDDING_FREQ_BLOCK
            )
            grad_inputs = grad_inputs / freq.to(torch.float32).unsqueeze(1)

    return grad_inputs.to(dtype) if acc_dtype != dtype else grad_inputs


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    """Embedding backward: accumulate grad rows back into the rows of
    grad_inputs.

    The main path forwards to the native aten implementation (0.43ms on
    gcu400; this backend's Triton atomic_add needs 35ms). Once
    aten::embedding_backward is taken over by flag_gems (enable scenario) the
    forwarding reenters this function, which then uses the atomic fallback so
    it neither recurses nor returns wrong results.
    """
    logger.debug("GEMS_ENFLAME EMBEDDING_BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    grad_outputs = grad_outputs.contiguous()
    indices = indices.contiguous()
    # topsaten only accepts int32 index tensors: an int64 index tensor aborts
    # inside the native kernel (TOPSATEN_STATUS_BAD_PARAM, "index tensor must be
    # of int32 data type") instead of raising a Python exception, which kills the
    # process. Cast here so the int64-enabled default configuration stays usable.
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    if getattr(_NATIVE_BACKWARD_STATE, "forwarding", False):
        return _embedding_backward_atomic(
            grad_outputs, indices, num_weights, padding_idx, scale_grad_by_freq
        )

    # aten uses -1 for "no padding_idx"; the Python side also accepts None.
    normalized_padding_idx = -1 if padding_idx is None else padding_idx
    _NATIVE_BACKWARD_STATE.forwarding = True
    try:
        python_key = torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
        with torch._C._ExcludeDispatchKeyGuard(python_key):
            return torch.ops.aten.embedding_backward(
                grad_outputs,
                indices,
                num_weights,
                normalized_padding_idx,
                scale_grad_by_freq,
                sparse,
            )
    finally:
        _NATIVE_BACKWARD_STATE.forwarding = False
