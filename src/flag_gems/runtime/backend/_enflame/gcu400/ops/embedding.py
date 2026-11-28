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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Column limit of one tile; for larger N the kernel loops over columns.
_EMBEDDING_BLOCK_MAX = 4096

# The device runtime stores int64 tensors either natively or as a narrowed int32
# buffer, depending on the TORCH_GCU_ENABLE_INT64_AND_UINT64 switch. Only the
# native mode keeps the low int32 word at a stride of two elements, so the
# low-word read below is only valid there; in the narrowed mode the index must
# be cast, otherwise every gathered row is taken from garbage.
_NATIVE_INT64 = os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1"


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
    FULL_TILE: tl.constexpr,
):
    """Copy whole rows of ``weight`` into the output following ``indices``.

    Two bad paths must be avoided on gcu400: reading int64 indices on the
    device returns wrong values (the whole gather is misaligned), and a scalar
    (0-d) index in pointer arithmetic aborts with SIGABRT. The indices are
    therefore normalized to int32 on the host, the kernel keeps a ``[ROWS]``
    vector index, and all pointer arithmetic stays 2D.

    ``FULL_TILE`` means the tile divides both ``M`` and ``N`` exactly, so no
    bound mask is generated at all. That is the dominant win of this kernel:
    the masked form measured 7.7/8.0/17.0 us on the 128/256/1024 shapes while
    the unmasked form reaches 4.8/4.8/12.2 us (fp16, GCU400, do_bench median),
    because the masked 2D tile cannot be lowered to a full-width transfer.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    if IDX_I64:
        # int64 indices are read as their low int32 word instead of running a separate
        # cast kernel on the host: any embedding index fits in int32, so the low word is
        # the value, and this removes one kernel launch (~4.5 us) from every call, which
        # dominates the latency of the small shapes in this op's benchmark.
        row_index_ptr = index_ptr + rows * 2
    else:
        row_index_ptr = index_ptr + rows
    if FULL_TILE:
        src_rows = tl.load(row_index_ptr)
        for col_start in tl.static_range(0, N, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            value = tl.load(weight_ptr + src_rows[:, None] * N + cols[None, :])
            tl.store(out_ptr + rows[:, None] * N + cols[None, :], value)
    else:
        row_mask = rows < M
        src_rows = tl.load(row_index_ptr, mask=row_mask, other=0)
        for col_start in range(0, N, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (cols[None, :] < N)
            value = tl.load(
                weight_ptr + src_rows[:, None] * N + cols[None, :], mask=mask, other=0.0
            )
            tl.store(out_ptr + rows[:, None] * N + cols[None, :], value, mask=mask)


def _embedding_tile(N, element_size, M):
    """Pick (ROWS, BLOCK_N, num_warps) for this shape.

    The tiers come from a tile/warp sweep over the five benchmark shapes on
    GCU400. Narrow rows cannot fill the vector units (one row per program
    measured 6.6 us on every small shape), so the row count grows with the row
    width: 16 rows up to 256 columns, then a 512/1024-column block for the wide
    shapes. The wide tier deliberately avoids num_warps=2 with BLOCK_N=512:
    that combination measured 20-138 us instead of ~11 us, an unstable
    scheduling regime rather than a small regression.
    """
    if N <= 256:
        rows, block_n, warps = 16, triton.next_power_of_2(N), 1
    elif element_size <= 2:
        rows, block_n, warps = 64, 1024, 4
    else:
        rows, block_n, warps = 32, 512, 4
    block_n = min(block_n, _EMBEDDING_BLOCK_MAX)
    rows = max(1, min(rows, M))
    return rows, block_n, warps


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
    # int64 indices are passed through as an int32 view (low word per element)
    # instead of being cast on the device: the cast costs a full extra kernel
    # launch, and reading int64 lanes directly is unreliable on gcu400. That is
    # only valid with native int64 storage; with the runtime switch off the
    # device keeps int64 indices in a narrowed int32 buffer, where the low word
    # of the next element is garbage, so the index is cast instead.
    index_is_i64 = gather_index.dtype == torch.int64
    if index_is_i64 and _NATIVE_INT64:
        gather_index = gather_index.view(torch.int32)
    elif gather_index.dtype != torch.int32:
        gather_index = gather_index.to(torch.int32)
        index_is_i64 = False

    rows, block_n, num_warps = _embedding_tile(N, weight.element_size(), M)
    # The mask-free path is only exact when the tile divides the problem; the
    # wide tier keeps BLOCK_N below N, so an exact column split is required too.
    full_tile = (M % rows == 0) and (N % block_n == 0)
    grid = (triton.cdiv(M, rows),)
    with torch_device_fn.device(weight.device):
        _embedding_gather_kernel[grid](
            output,
            gather_index,
            weight,
            M,
            N,
            rows,
            block_n,
            index_is_i64,
            full_tile,
            num_warps=num_warps,
        )

    return output


# ---------------------------------------------------------------------------
# Backward: rows of grad_output are accumulated into grad_input rows with
# atomic_add. The atomic formulation is bit-exact against the CPU reference for
# every padding_idx / scale_grad_by_freq combination; it is the only backward
# path (no native aten forwarding).
# ---------------------------------------------------------------------------
_EMBEDDING_BACKWARD_ROWS = 4
_EMBEDDING_FREQ_BLOCK = 256


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


@triton.jit
def _embedding_backward_scale_kernel(
    rows_ptr,
    freq_ptr,
    num_rows,
    N,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Divide each accumulated row by the number of times its index occurs.

    The frequency is clamped to >= 1 in-kernel: rows that never occur keep an
    all-zero gradient, so dividing by the raw zero count would yield NaN.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < num_rows
    freq = tl.maximum(tl.load(freq_ptr + rows, mask=row_mask, other=1), 1)
    inverse_freq = 1.0 / freq.to(tl.float32)
    for col_start in range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        ptr = rows_ptr + rows[:, None] * N + cols[None, :]
        value = tl.load(ptr, mask=mask, other=0.0)
        tl.store(ptr, value * inverse_freq[:, None], mask=mask)


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
            _embedding_backward_scale_kernel[
                (triton.cdiv(num_weights, _EMBEDDING_BACKWARD_ROWS),)
            ](
                grad_inputs,
                freq,
                num_weights,
                N,
                _EMBEDDING_BACKWARD_ROWS,
                block_n,
            )

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

    The whole computation runs in the Triton atomic-accumulate kernel below;
    forwarding to ``aten::embedding_backward`` is not used because the target
    operator must be computed by the Triton implementation. The atomic path is
    bit-exact against the CPU reference for fp32/fp16/bf16 and every
    padding_idx / scale_grad_by_freq combination.
    """
    logger.debug("GEMS_ENFLAME EMBEDDING_BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    grad_outputs = grad_outputs.contiguous()
    indices = indices.contiguous()
    return _embedding_backward_atomic(
        grad_outputs, indices, num_weights, padding_idx, scale_grad_by_freq
    )
