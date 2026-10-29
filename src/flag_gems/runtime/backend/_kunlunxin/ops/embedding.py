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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Rows narrower than this are gathered by the flat kernel (one program per
# BLOCK_SIZE output elements); wider rows keep the per-row kernel, which is
# already fully contiguous at that width.
FLAT_MAX_N = 4096
# Per-program tile for the flat path. Preferred sizes in descending order;
# the launcher picks the first that divides the output exactly so the kernel
# can skip masking (worth more than the ideal tile width -- see EXACT in
# embedding_kernel_flat). Measured unmasked sp on fp32/fp16:
#   128x128    blk 4096  -> 1.02/1.01
#   256x256    blk 8192  -> 1.00/0.99
#   1024x1024  blk 32768 -> 0.55/0.69
# Tiles below 2048 are never competitive (grid explodes: 256-wide on
# 1024x1024 measured sp 0.011), hence the floor.
FLAT_BLOCK_CANDIDATES = (32768, 16384, 8192, 4096, 2048)
FLAT_BLOCK_MIN = 2048
# Outputs this small keep the masked path. The unmasked kernel threw a device
# "kernel exception" on a 16-element output on one card (and the same case ran
# clean on another), so the tiny end is not worth the risk: at that size torch
# finishes in ~5us, which is under the Triton launch floor, so no tiling can
# win there anyway.
FLAT_NOMASK_MIN_NUMEL = 1024
# The XPU default (512) is slower here and is known to miscompute wide reduces
# elsewhere in this backend; keep the tuned value.
BUFFER_SIZE_LIMIT = 2048


@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


@libentry()
@triton.jit
def embedding_kernel_flat(
    out_ptr,
    in_ptr,
    weight_ptr,
    NUMEL,  # M * N
    N: tl.constexpr,  # embedding_dim
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,  # NUMEL % BLOCK_SIZE == 0 -> no mask needed
):
    # Flat variant of embedding_kernel: one program per BLOCK_SIZE *elements*
    # of the output instead of one program per row.
    #
    # The per-row kernel ties the tile width to N, so for the benchmarked
    # shapes (N = 4..1024) every program moves at most one short row while the
    # grid grows to M programs. That is launch-bound: (256,256) fp32 measured
    # 81.9us per-row vs 18.9us flat, and (128,128) 50.6us vs 18.5us.
    #
    # Here the tile stays wide regardless of N, the store is fully contiguous,
    # and only the gathered load is strided.
    #
    # EXACT matters a lot: masked accesses take a much slower path on this
    # backend, so when the tile divides the output evenly we emit none. On
    # (1024,1024) that is sp 0.32 -> 0.55 (fp32) and 0.30 -> 0.69 (fp16) at
    # the same tile size -- a bigger win than any block-size choice.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EXACT:
        row_idx = tl.load(in_ptr + offset // N)
        embedding_weight = tl.load(weight_ptr + row_idx * N + offset % N)
        tl.store(out_ptr + offset, embedding_weight)
    else:
        # Tail tile. A masked load does NOT reliably honour other= on this
        # backend (it can return garbage), and a garbage row_idx would make
        # row_idx * N an out-of-range address -- an illegal access, not just a
        # wrong value. So CLAMP the gather offset into range and mask only the
        # store, which decides what is actually written.
        mask = offset < NUMEL
        safe = tl.minimum(offset, NUMEL - 1)
        row_idx = tl.load(in_ptr + safe // N)
        embedding_weight = tl.load(weight_ptr + row_idx * N + safe % N)
        tl.store(out_ptr + offset, embedding_weight, mask=mask)


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

    index_element = tl.load(indices + offsets, mask=mask)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel(
    grad_in,  # pointer to the gradient input
    grad_out,  # pointer to the gradient output
    indices,  # pointer to the input
    padding_idx,  # padding_idx
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_in += row_idx * N
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
        if tl.constexpr(embedding_grad.dtype.is_bf16()):
            embedding_grad = embedding_grad.to(tl.float32)
        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_in += row_idx * N
            embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
            if tl.constexpr(embedding_grad.dtype.is_bf16()):
                embedding_grad = embedding_grad.to(tl.float32)
            tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def embedding_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = ext.program_id(0)
    row_step = ext.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask)


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    logger.debug("GEMS_KUNLUNXIN EMBEDDING")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]

    # TODO: remove contiguous enforcement
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    numel = M * N
    with torch_device_fn.device(weight.device):
        if numel > 0 and N < FLAT_MAX_N:
            # Narrow rows: a per-row grid is launch-bound, so go flat.
            #
            # Prefer the largest candidate tile that divides numel exactly,
            # which lets the kernel drop masking entirely. Never exceed numel:
            # a mostly-masked wide tile is not just wasteful, it takes minutes
            # to compile on this backend.
            cap = min(triton.next_power_of_2(numel), FLAT_BLOCK_CANDIDATES[0])
            block_size = min(cap, FLAT_BLOCK_MIN)
            for cand in FLAT_BLOCK_CANDIDATES:
                if cand <= cap and numel % cand == 0:
                    block_size = cand
                    break
            exact = numel % block_size == 0 and numel >= FLAT_NOMASK_MIN_NUMEL
            grid = (triton.cdiv(numel, block_size),)
            embedding_kernel_flat[grid](
                output,
                indices,
                weight,
                numel,
                N,
                block_size,
                exact,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
            )
        else:
            # Wide rows already give each program a full tile of work, and the
            # per-row kernel keeps both load and store contiguous.
            embedding_kernel[M,](
                output, indices, weight, N, triton.next_power_of_2(N)
            )

    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS_KUNLUNXIN EMBEDDING_BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = grad_outputs.shape[-1]

    grad_inputs = torch.zeros(
        (num_weights, grad_outputs.shape[-1]),
        device=grad_outputs.device,
        dtype=(
            torch.float32
            if grad_outputs.dtype is torch.bfloat16
            else grad_outputs.dtype
        ),
    )

    if scale_grad_by_freq:
        indice_freq = torch.zeros(
            (num_weights,),
            requires_grad=False,
            device=grad_outputs.device,
            dtype=torch.int32,
        )
        INDICE_BLOCK_SIZE = 256
        indice_grid = (triton.cdiv(M, INDICE_BLOCK_SIZE),)

        with torch_device_fn.device(grad_outputs.device):
            indice_freq_kernel[indice_grid](
                indice_freq,
                indices,
                M,
                INDICE_BLOCK_SIZE,
                isCLOSE_TTXPU_O_ATOMIC_SIM=True,
            )
    else:
        indice_freq = None

    BLOCK_SIZE = triton.next_power_of_2(N)

    HAS_PADDING_IDX = padding_idx is not None

    with torch_device_fn.device(grad_outputs.device):
        embedding_backward_kernel[M,](
            grad_inputs,
            grad_outputs,
            indices,
            padding_idx,
            HAS_PADDING_IDX,
            N,
            BLOCK_SIZE,
        )

    if scale_grad_by_freq:
        with torch_device_fn.device(grad_outputs.device):
            embedding_grad_scale_kernel[M,](
                grad_inputs, indice_freq, num_weights, N, BLOCK_SIZE
            )
    return (
        grad_inputs.to(torch.bfloat16)
        if grad_outputs.dtype is torch.bfloat16
        else grad_inputs
    )
