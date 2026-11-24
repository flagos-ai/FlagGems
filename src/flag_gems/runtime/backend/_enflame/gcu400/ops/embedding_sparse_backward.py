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

import flag_gems

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per optimization round; only affects the device where sparse output is materialized, not numeric semantics.
USE_CPU_SPARSE = 0
# Bytes moved per program. values and grad_output are both contiguous (N, D) memory, and only a
# flat 1-D view lets the compiler emit contiguous accesses; a 2-D tile was measured to be split
# into per-row small transactions.
# Fixed in bytes rather than elements. The 8KB tier is fastest (fp16 0.94ms) but the generated
# kernel text is ~405KB, and loading the device program inside a process that already holds many
# modules (pytest) fails with loadSegments alloc mem abort; the 4KB tier (fp16 1.27ms, fp32
# 1.27ms) loads reliably and is still 3.7x faster than the original per-row transactions.
BYTES_PER_PROG = 4096
# Element-count floor, so a large element_size cannot degrade the block below one vector width.
MIN_BLOCK_ELEMS = 256
# Hard grid.x cap: beyond it more elements are packed into the same program.
GRID_X_LIMIT = 65535
# --- TUNING PARAMS END ---


@triton.jit
def _masked_copy_flat_kernel(
    grad_output_ptr,  # (N, D) original gradient rows (contiguous)
    indices_ptr,  # (N,) row ids (int32, normalized on the host)
    values_ptr,  # (N, D) output values (contiguous)
    padding_idx,
    total,  # N * D
    EMBED_DIM: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    # Flat view: one program moves BLOCK_ELEMS consecutive elements, the row id comes from
    # offs // EMBED_DIM (a power-of-two width compiles into a shift), and only rows with
    # index == padding_idx are zeroed while every other element is copied as is. This way the
    # memory transactions of one program are BLOCK_ELEMS consecutive elements (4-8KB) instead of
    # a fragmented "256B per row" pattern.
    pid = tl.program_id(0)
    offs = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offs < total
    row = offs // EMBED_DIM
    # Normalize to i32: the host already converted, this defensive cast tolerates both int64 layouts.
    # Row ids repeat heavily inside one program (only EMBED_DIM distinct addresses), so it is cache friendly.
    row_idx = tl.load(indices_ptr + row, mask=mask, other=0).to(tl.int32)
    keep = row_idx != padding_idx
    grad_tile = tl.load(grad_output_ptr + offs, mask=mask, other=0)
    tl.store(values_ptr + offs, tl.where(keep, grad_tile, 0.0), mask=mask)


def _assemble_device_coo(
    sparse_indices: torch.Tensor,
    values: torch.Tensor,
    num_weights: int,
    embed_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Assemble the unmerged ``(num_weights, embed_dim)`` COO on the device.

    ``torch.sparse_coo_tensor`` (the python factory) keeps one values-sized buffer
    alive per call on this backend: measured growth is exactly
    ``values.numel() * itemsize`` per call and neither ``gc.collect()`` nor
    ``torch.cuda.empty_cache()`` reclaims it, so a benchmark loop that repeats the
    op a few hundred times exhausts device memory.  The underlying aten factory
    takes the already materialised values and keeps no reference, so it is used
    directly here.  The COO stays uncoalesced, matching aten's sparse backward
    (one entry per grad row, duplicates not merged).
    """
    if sparse_indices.dtype != torch.int64:
        sparse_indices = sparse_indices.to(torch.int64)
    return torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        1,  # sparse dims: the embedding-table row id
        1,  # dense dims: the embedding axis
        [num_weights, embed_dim],
        sparse_indices,
        values,
        is_coalesced=False,
        dtype=dtype,
        layout=torch.sparse_coo,
        device=device,
    )


def embedding_sparse_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    logger.debug("GEMS_ENFLAME EMBEDDING_SPARSE_BACKWARD")
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    if (
        grad_output.device.type != flag_gems.device
        or indices.device.type != flag_gems.device
        or grad_output.device != indices.device
    ):
        raise ValueError(
            f"Inputs must be {flag_gems.device} tensors on the same device."
        )
    if scale_grad_by_freq:
        # Matches aten: frequency scaling is not defined for sparse gradients.
        raise RuntimeError(
            "embedding_backward: scale_grad_by_freq not supported with sparse gradients"
        )

    device = grad_output.device
    assert (
        grad_output.dim() >= 2
    ), "grad_output must have embedding dimension as the last dim."

    D = grad_output.shape[-1]
    go = grad_output.contiguous().view(-1, D)  # (N, D)
    idx = indices.contiguous().view(-1)  # (N,)
    N = idx.numel()
    assert go.shape[0] == N, "indices number must match grad_output rows."

    # aten sparse backward emits one COO entry per grad row, dropping rows whose index is
    # padding_idx and never merging duplicate coordinates: the result is an unmerged (num_weights, D) COO.
    has_padding = padding_idx is not None and padding_idx >= 0
    # P21: GCU has no sparse backend, so the output returns as a CPU-side COO (as in the aten CPU fallback).
    use_cpu_staging = USE_CPU_SPARSE and device.type != "cpu"
    # The COO indices are row ids that take no part in device computation, so the original dtype is kept:
    # the int64 -> int32 normalization only runs when a kernel needs the row ids (padding branch),
    # avoiding a wasted cast kernel when there is no padding.
    if not has_padding and not use_cpu_staging and idx.dtype != torch.int64:
        idx = idx.to(torch.int64)

    if use_cpu_staging:
        idx_cpu = idx.detach().cpu().contiguous()
        sparse_indices = idx_cpu.view(1, -1)
        # With no padding every row appears in the output, so grad_output rows are
        # the COO values and no extra buffer is needed.
        values = go.detach().cpu()
        if has_padding:
            # Under to_dense semantics "drop padding rows" equals "zero those rows"; zeroing by explicit
            # row id is over 4x faster than a boolean-mask index (0.13ms vs 0.65ms @65536x128 fp16).
            pad_rows = (idx_cpu == padding_idx).nonzero().flatten()
            if pad_rows.numel():
                values[pad_rows] = 0
        out = torch.sparse_coo_tensor(
            sparse_indices,
            values,
            (num_weights, D),
            dtype=grad_output.dtype,
            device="cpu",
        )
    else:
        if has_padding:
            # P17: on some torch_gcu layouts int64 indices are packed into 4B storage, and an 8B-stride
            # device read returns interleaved wrong values; normalize to real int32 before the kernel reads.
            if idx.dtype != torch.int32:
                idx = idx.to(torch.int32)
            sparse_indices = idx.view(1, -1)
            # The device-side COO must stay on GCU: full-row copy plus zeroing the padding rows.
            values = torch.empty((N, D), device=device, dtype=grad_output.dtype)
            if N > 0:
                total = N * D
                # Convert the pixel width to an element count so the moved byte count matches across dtypes.
                block_elems = max(
                    MIN_BLOCK_ELEMS,
                    BYTES_PER_PROG // grad_output.element_size(),
                )
                # grid.x hard cap 65535: grow the per-program transfer automatically for large element counts.
                block_elems = max(
                    block_elems,
                    triton.next_power_of_2(triton.cdiv(total, GRID_X_LIMIT)),
                )
                _masked_copy_flat_kernel[(triton.cdiv(total, block_elems),)](
                    go,
                    idx,
                    values,
                    padding_idx,
                    total,
                    EMBED_DIM=D,
                    BLOCK_ELEMS=block_elems,
                )
        else:
            sparse_indices = idx.view(1, -1)
            values = go
        out = _assemble_device_coo(
            sparse_indices, values, num_weights, D, grad_output.dtype, device
        )
    return out
