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

"""Pure Triton implementation of ``aten::rnn_tanh.input``.

The implementation deliberately does not compose the recurrent operation from
PyTorch operators.  Forward recurrent steps, BPTT, input/weight gradients,
layout conversion, dropout, and zero filling are all Triton kernels.  PyTorch
is used only for tensor allocation and the ``autograd.Function`` boundary.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.tanh import tanh_kernel
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
flag_gems_tanh_scalar = tanh_kernel._scalar_fn

_CHUNK = tl.constexpr(32)
_DOT_PRECISION = tl.constexpr(
    "tf32x3" if runtime.device.vendor_name == "nvidia" else "ieee"
)


@triton.jit
def _load_input(
    input_ptr,
    t,
    b_offs,
    i_offs,
    batch_size,
    input_size,
    stride_t,
    stride_b,
    stride_i,
    dropout_p,
    dropout_seed,
    dropout_base,
    APPLY_DROPOUT: tl.constexpr,
):
    mask = (b_offs[:, None] < batch_size) & (i_offs[None, :] < input_size)
    offsets = t * stride_t + b_offs[:, None] * stride_b + i_offs[None, :] * stride_i
    value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    if APPLY_DROPOUT:
        random_offsets = (
            dropout_base
            + (t * batch_size + b_offs[:, None]) * input_size
            + i_offs[None, :]
        )
        keep = tl.rand(dropout_seed, random_offsets) > dropout_p
        value = tl.where(keep & mask, value / (1.0 - dropout_p), 0.0)
    return value


@libentry()
@triton.jit
def rnn_tanh_forward_dot_kernel(
    input_ptr,
    hx_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_ptr,
    seq_len,
    batch_size,
    input_size,
    hidden_size,
    input_stride_t,
    input_stride_b,
    input_stride_i,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    weight_ih_stride_h,
    weight_ih_stride_i,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    bias_ih_stride,
    bias_hh_stride,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    dropout_p,
    dropout_seed,
    dropout_base,
    HAS_BIAS: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Persistent tensor-core friendly RNN kernel for moderate feature sizes."""
    batch_start = tl.program_id(0) * BLOCK_B
    b_offs = batch_start + tl.arange(0, BLOCK_B)
    i_offs = tl.arange(0, BLOCK_I)
    h_offs = tl.arange(0, BLOCK_H)
    b_mask = b_offs < batch_size
    h_mask = h_offs < hidden_size

    hx_offsets = (
        state_index * hx_stride_state
        + b_offs[:, None] * hx_stride_b
        + h_offs[None, :] * hx_stride_h
    )
    hidden = tl.load(
        hx_ptr + hx_offsets,
        mask=b_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    w_ih_offsets = (
        h_offs[:, None] * weight_ih_stride_h + i_offs[None, :] * weight_ih_stride_i
    )
    weight_ih = tl.load(
        weight_ih_ptr + w_ih_offsets,
        mask=(h_offs[:, None] < hidden_size) & (i_offs[None, :] < input_size),
        other=0.0,
    )
    w_hh_offsets = (
        h_offs[:, None] * weight_hh_stride_h0 + h_offs[None, :] * weight_hh_stride_h1
    )
    weight_hh = tl.load(
        weight_hh_ptr + w_hh_offsets,
        mask=h_mask[:, None] & h_mask[None, :],
        other=0.0,
    )

    if HAS_BIAS:
        bias = tl.load(
            bias_ih_ptr + h_offs * bias_ih_stride,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        bias += tl.load(
            bias_hh_ptr + h_offs * bias_hh_stride,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_H], dtype=tl.float32)

    for step in range(seq_len):
        t = step if direction == 0 else seq_len - 1 - step
        x = _load_input(
            input_ptr,
            t,
            b_offs,
            i_offs,
            batch_size,
            input_size,
            input_stride_t,
            input_stride_b,
            input_stride_i,
            dropout_p,
            dropout_seed,
            dropout_base,
            APPLY_DROPOUT,
        )
        # tl.dot accumulates into fp32 for fp16/bf16 inputs and uses the native
        # NVIDIA matrix path.  Masked padding makes arbitrary feature sizes safe.
        # Keep reduced-precision inputs in their storage dtype so NVIDIA tensor
        # cores are used for both fp16 and bf16.  Accumulation remains fp32.
        input_linear = tl.dot(
            x.to(weight_ih.dtype), tl.trans(weight_ih), input_precision=_DOT_PRECISION
        ).to(tl.float32)
        hidden_linear = tl.dot(
            hidden.to(weight_hh.dtype),
            tl.trans(weight_hh),
            input_precision=_DOT_PRECISION,
        ).to(tl.float32)
        hidden = tl_extra_shim.tanh(input_linear + hidden_linear + bias[None, :])
        hidden = tl.where(b_mask[:, None] & h_mask[None, :], hidden, 0.0)

        output_offsets = (
            (t * batch_size + b_offs[:, None]) * output_feature_size
            + direction * hidden_size
            + h_offs[None, :]
        )
        tl.store(
            output_ptr + output_offsets,
            hidden,
            mask=b_mask[:, None] & h_mask[None, :],
        )

    hidden_offsets = (
        state_index * batch_size + b_offs[:, None]
    ) * hidden_size + h_offs[None, :]
    tl.store(
        hidden_ptr + hidden_offsets,
        hidden,
        mask=b_mask[:, None] & h_mask[None, :],
    )


@libentry()
@triton.jit
def rnn_tanh_input_linear_kernel(
    input_ptr,
    weight_ih_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    linear_ptr,
    rows,
    batch_size,
    input_size,
    hidden_size,
    input_stride_t,
    input_stride_b,
    input_stride_i,
    weight_ih_stride_h,
    weight_ih_stride_i,
    bias_ih_stride,
    bias_hh_stride,
    dropout_p,
    dropout_seed,
    dropout_base,
    HAS_BIAS: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Precompute all input-to-hidden products with backend-friendly tiles."""
    row_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    row_mask = row_offs < rows
    n_mask = n_offs < hidden_size
    t = row_offs // batch_size
    b = row_offs - t * batch_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, input_size, BLOCK_K):
        ks = k_start + k_offs
        k_mask = ks < input_size
        input_offsets = (
            t[:, None] * input_stride_t
            + b[:, None] * input_stride_b
            + ks[None, :] * input_stride_i
        )
        x = tl.load(
            input_ptr + input_offsets,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if APPLY_DROPOUT:
            random_offsets = dropout_base + row_offs[:, None] * input_size + ks[None, :]
            keep = tl.rand(dropout_seed, random_offsets) > dropout_p
            x = tl.where(
                keep & row_mask[:, None] & k_mask[None, :],
                x / (1.0 - dropout_p),
                0.0,
            )
        weight = tl.load(
            weight_ih_ptr
            + ks[:, None] * weight_ih_stride_i
            + n_offs[None, :] * weight_ih_stride_h,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x.to(weight.dtype), weight, input_precision=_DOT_PRECISION)

    if HAS_BIAS:
        bias = tl.load(
            bias_ih_ptr + n_offs * bias_ih_stride,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        bias += tl.load(
            bias_hh_ptr + n_offs * bias_hh_stride,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        linear_ptr + row_offs[:, None] * hidden_size + n_offs[None, :],
        acc,
        mask=row_mask[:, None] & n_mask[None, :],
    )


@libentry()
@triton.jit
def rnn_tanh_recurrent_ascend_kernel(
    hx_ptr,
    weight_hh_ptr,
    bias_hh_ptr,
    linear_ptr,
    work_ptr,
    output_ptr,
    hidden_ptr,
    time_index,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    bias_hh_stride,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    FIRST_STEP: tl.constexpr,
    FINAL_STEP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Run one synchronized recurrent step using small Ascend matmul tiles."""
    m_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    m_mask = m_offs < batch_size
    n_mask = n_offs < hidden_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        ks = k_start + k_offs
        k_mask = ks < hidden_size
        if FIRST_STEP:
            previous = tl.load(
                hx_ptr
                + state_index * hx_stride_state
                + m_offs[:, None] * hx_stride_b
                + ks[None, :] * hx_stride_h,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
        else:
            previous_time = time_index - 1 if direction == 0 else time_index + 1
            previous = tl.load(
                output_ptr
                + (previous_time * batch_size + m_offs[:, None]) * output_feature_size
                + direction * hidden_size
                + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
        weight = tl.load(
            weight_hh_ptr
            + ks[:, None] * weight_hh_stride_h1
            + n_offs[None, :] * weight_hh_stride_h0,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(previous.to(weight.dtype), weight, input_precision=_DOT_PRECISION)

    row = time_index * batch_size + m_offs
    acc += tl.load(
        linear_ptr + row[:, None] * hidden_size + n_offs[None, :],
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    current = flag_gems_tanh_scalar(acc)
    mask = m_mask[:, None] & n_mask[None, :]
    output_offsets = (
        row[:, None] * output_feature_size + direction * hidden_size + n_offs[None, :]
    )
    tl.store(output_ptr + output_offsets, current, mask=mask)
    if FINAL_STEP:
        hidden_offsets = (
            state_index * batch_size + m_offs[:, None]
        ) * hidden_size + n_offs[None, :]
        tl.store(hidden_ptr + hidden_offsets, current, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_recurrent_addmm_ascend_kernel(
    hx_ptr,
    weight_hh_ptr,
    linear_ptr,
    output_ptr,
    time_index,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    FIRST_STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one recurrent addmm; FlagGems tanh runs separately in-place."""
    m_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    m_mask = m_offs < batch_size
    n_mask = n_offs < hidden_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        ks = k_start + k_offs
        k_mask = ks < hidden_size
        if FIRST_STEP:
            previous = tl.load(
                hx_ptr
                + state_index * hx_stride_state
                + m_offs[:, None] * hx_stride_b
                + ks[None, :] * hx_stride_h,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
        else:
            previous_time = time_index - 1 if direction == 0 else time_index + 1
            previous = tl.load(
                output_ptr
                + (previous_time * batch_size + m_offs[:, None]) * output_feature_size
                + direction * hidden_size
                + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
        weight = tl.load(
            weight_hh_ptr
            + ks[:, None] * weight_hh_stride_h1
            + n_offs[None, :] * weight_hh_stride_h0,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(previous.to(weight.dtype), weight, input_precision=_DOT_PRECISION)

    row = time_index * batch_size + m_offs
    acc += tl.load(
        linear_ptr + row[:, None] * hidden_size + n_offs[None, :],
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    output_offsets = (
        row[:, None] * output_feature_size + direction * hidden_size + n_offs[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@libentry()
@triton.jit
def rnn_tanh_activation_ascend_kernel(
    output_ptr,
    hidden_ptr,
    time_index,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    FINAL_STEP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply the existing FlagGems tanh scalar and optionally save final state."""
    m_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (m_offs[:, None] < batch_size) & (n_offs[None, :] < hidden_size)
    row = time_index * batch_size + m_offs
    output_offsets = (
        row[:, None] * output_feature_size + direction * hidden_size + n_offs[None, :]
    )
    current = tl.load(output_ptr + output_offsets, mask=mask, other=0.0)
    current = flag_gems_tanh_scalar(current.to(tl.float32))
    tl.store(output_ptr + output_offsets, current, mask=mask)
    if FINAL_STEP:
        hidden_offsets = (
            state_index * batch_size + m_offs[:, None]
        ) * hidden_size + n_offs[None, :]
        tl.store(hidden_ptr + hidden_offsets, current, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_recurrent_chunk_ascend_kernel(
    hx_ptr,
    weight_hh_ptr,
    linear_ptr,
    output_ptr,
    hidden_ptr,
    start_step,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    output_feature_size,
    state_index,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    direction: tl.constexpr,
    FIRST_CHUNK: tl.constexpr,
    FINAL_CHUNK: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Run a short recurrence chunk using the existing FlagGems tanh scalar."""
    b_offs = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    h_offs = tl.arange(0, BLOCK_H)
    mask = (b_offs[:, None] < batch_size) & (h_offs[None, :] < hidden_size)
    start_time = start_step if direction == 0 else seq_len - 1 - start_step
    if FIRST_CHUNK:
        hidden = tl.load(
            hx_ptr
            + state_index * hx_stride_state
            + b_offs[:, None] * hx_stride_b
            + h_offs[None, :] * hx_stride_h,
            mask=mask,
            other=0.0,
        )
    else:
        previous_time = start_time - 1 if direction == 0 else start_time + 1
        hidden = tl.load(
            output_ptr
            + (previous_time * batch_size + b_offs[:, None]) * output_feature_size
            + direction * hidden_size
            + h_offs[None, :],
            mask=mask,
            other=0.0,
        )
    weight_hh = tl.load(
        weight_hh_ptr
        + h_offs[:, None] * weight_hh_stride_h0
        + h_offs[None, :] * weight_hh_stride_h1,
        mask=(h_offs[:, None] < hidden_size) & (h_offs[None, :] < hidden_size),
        other=0.0,
    )
    for local_step in range(STEPS):
        logical_step = start_step + local_step
        time_index = logical_step if direction == 0 else seq_len - 1 - logical_step
        input_linear = tl.load(
            linear_ptr
            + (time_index * batch_size + b_offs[:, None]) * hidden_size
            + h_offs[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        hidden_linear = tl.dot(hidden, tl.trans(weight_hh)).to(tl.float32)
        hidden = flag_gems_tanh_scalar(input_linear + hidden_linear).to(weight_hh.dtype)
        hidden = tl.where(mask, hidden, 0.0).to(weight_hh.dtype)
        tl.store(
            output_ptr
            + (time_index * batch_size + b_offs[:, None]) * output_feature_size
            + direction * hidden_size
            + h_offs[None, :],
            hidden,
            mask=mask,
        )
    if FINAL_CHUNK:
        tl.store(
            hidden_ptr
            + (state_index * batch_size + b_offs[:, None]) * hidden_size
            + h_offs[None, :],
            hidden,
            mask=mask,
        )


@libentry()
@triton.jit
def rnn_tanh_recurrent_persistent_kernel(
    hx_ptr,
    weight_hh_ptr,
    linear_ptr,
    output_ptr,
    hidden_ptr,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Consume a precomputed input projection in one recurrent launch.

    Keeping the recurrent state and recurrent weight resident avoids one kernel
    launch per time step.  The input projection remains a separate, highly
    parallel matrix multiplication, so it is not serialized inside the
    persistent recurrence.
    """
    batch_start = tl.program_id(0) * BLOCK_B
    b_offs = batch_start + tl.arange(0, BLOCK_B)
    h_offs = tl.arange(0, BLOCK_H)
    b_mask = b_offs < batch_size
    h_mask = h_offs < hidden_size
    mask = b_mask[:, None] & h_mask[None, :]

    hidden = tl.load(
        hx_ptr
        + state_index * hx_stride_state
        + b_offs[:, None] * hx_stride_b
        + h_offs[None, :] * hx_stride_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_hh = tl.load(
        weight_hh_ptr
        + h_offs[:, None] * weight_hh_stride_h0
        + h_offs[None, :] * weight_hh_stride_h1,
        mask=h_mask[:, None] & h_mask[None, :],
        other=0.0,
    )

    for step in range(seq_len):
        t = step if direction == 0 else seq_len - 1 - step
        input_linear = tl.load(
            linear_ptr
            + (t * batch_size + b_offs[:, None]) * hidden_size
            + h_offs[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        hidden_linear = tl.dot(
            hidden.to(weight_hh.dtype),
            tl.trans(weight_hh),
            input_precision=_DOT_PRECISION,
        ).to(tl.float32)
        hidden = tl_extra_shim.tanh(input_linear + hidden_linear)
        hidden = tl.where(mask, hidden, 0.0)
        tl.store(
            output_ptr
            + (t * batch_size + b_offs[:, None]) * output_feature_size
            + direction * hidden_size
            + h_offs[None, :],
            hidden,
            mask=mask,
        )

    tl.store(
        hidden_ptr
        + (state_index * batch_size + b_offs[:, None]) * hidden_size
        + h_offs[None, :],
        hidden,
        mask=mask,
    )


@libentry()
@triton.jit
def rnn_tanh_forward_vector_kernel(
    input_ptr,
    hx_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_ptr,
    hidden_read_ptr,
    seq_len,
    batch_size,
    input_size,
    hidden_size,
    input_stride_t,
    input_stride_b,
    input_stride_i,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    weight_ih_stride_h,
    weight_ih_stride_i,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    bias_ih_stride,
    bias_hh_stride,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    dropout_p,
    dropout_seed,
    dropout_base,
    HAS_BIAS: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Register-bounded fallback for large input or hidden dimensions."""
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    chunk_offs = tl.arange(0, _CHUNK)
    h_block_offs = tl.arange(0, BLOCK_H)
    num_h_blocks = tl.cdiv(hidden_size, BLOCK_H)
    hidden_base = (state_index * batch_size + batch_idx) * hidden_size
    read_base = batch_idx * hidden_size

    for h_block in range(num_h_blocks):
        h_offs = h_block * BLOCK_H + h_block_offs
        h_mask = h_offs < hidden_size
        initial = tl.load(
            hx_ptr
            + state_index * hx_stride_state
            + batch_idx * hx_stride_b
            + h_offs * hx_stride_h,
            mask=h_mask,
            other=0.0,
        )
        tl.store(hidden_ptr + hidden_base + h_offs, initial, mask=h_mask)

    for step in range(seq_len):
        t = step if direction == 0 else seq_len - 1 - step
        for h_block in range(num_h_blocks):
            h_offs = h_block * BLOCK_H + h_block_offs
            h_mask = h_offs < hidden_size
            previous = tl.load(
                hidden_ptr + hidden_base + h_offs, mask=h_mask, other=0.0
            )
            tl.store(hidden_read_ptr + read_base + h_offs, previous, mask=h_mask)

        for h_block in range(num_h_blocks):
            h_offs = h_block * BLOCK_H + h_block_offs
            h_mask = h_offs < hidden_size
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)

            for i_start in range(0, input_size, _CHUNK):
                i_offs = i_start + chunk_offs
                i_mask = i_offs < input_size
                x_offsets = (
                    t * input_stride_t
                    + batch_idx * input_stride_b
                    + i_offs * input_stride_i
                )
                x = tl.load(input_ptr + x_offsets, mask=i_mask, other=0.0).to(
                    tl.float32
                )
                if APPLY_DROPOUT:
                    random_offsets = (
                        dropout_base
                        + (t * batch_size + batch_idx) * input_size
                        + i_offs
                    )
                    keep = tl.rand(dropout_seed, random_offsets) > dropout_p
                    x = tl.where(keep & i_mask, x / (1.0 - dropout_p), 0.0)
                w_offsets = (
                    h_offs[:, None] * weight_ih_stride_h
                    + i_offs[None, :] * weight_ih_stride_i
                )
                w = tl.load(
                    weight_ih_ptr + w_offsets,
                    mask=h_mask[:, None] & i_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(w * x[None, :], axis=1)

            for j_start in range(0, hidden_size, _CHUNK):
                j_offs = j_start + chunk_offs
                j_mask = j_offs < hidden_size
                previous = tl.load(
                    hidden_read_ptr + read_base + j_offs,
                    mask=j_mask,
                    other=0.0,
                ).to(tl.float32)
                w_offsets = (
                    h_offs[:, None] * weight_hh_stride_h0
                    + j_offs[None, :] * weight_hh_stride_h1
                )
                w = tl.load(
                    weight_hh_ptr + w_offsets,
                    mask=h_mask[:, None] & j_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(w * previous[None, :], axis=1)

            if HAS_BIAS:
                acc += tl.load(
                    bias_ih_ptr + h_offs * bias_ih_stride,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                acc += tl.load(
                    bias_hh_ptr + h_offs * bias_hh_stride,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
            current = tl_extra_shim.tanh(acc)
            tl.store(hidden_ptr + hidden_base + h_offs, current, mask=h_mask)
            output_offsets = (
                (t * batch_size + batch_idx) * output_feature_size
                + direction * hidden_size
                + h_offs
            )
            tl.store(output_ptr + output_offsets, current, mask=h_mask)


@libentry()
@triton.jit
def rnn_tanh_bptt_dot_kernel(
    grad_output_ptr,
    grad_hidden_ptr,
    layer_output_ptr,
    weight_hh_ptr,
    dpre_ptr,
    grad_hx_ptr,
    seq_len,
    batch_size,
    hidden_size,
    grad_stride_t,
    grad_stride_b,
    grad_stride_h,
    output_feature_size,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    state_index,
    direction: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_start = tl.program_id(0) * BLOCK_B
    b_offs = batch_start + tl.arange(0, BLOCK_B)
    h_offs = tl.arange(0, BLOCK_H)
    mask = (b_offs[:, None] < batch_size) & (h_offs[None, :] < hidden_size)

    whh_offsets = (
        h_offs[:, None] * weight_hh_stride_h0 + h_offs[None, :] * weight_hh_stride_h1
    )
    weight_hh = tl.load(
        weight_hh_ptr + whh_offsets,
        mask=(h_offs[:, None] < hidden_size) & (h_offs[None, :] < hidden_size),
        other=0.0,
    )
    grad_hidden_offsets = (
        state_index * batch_size + b_offs[:, None]
    ) * hidden_size + h_offs[None, :]
    dh = tl.load(grad_hidden_ptr + grad_hidden_offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    for step in range(seq_len):
        t = seq_len - 1 - step if direction == 0 else step
        go_offsets = (
            t * grad_stride_t
            + b_offs[:, None] * grad_stride_b
            + (direction * hidden_size + h_offs[None, :]) * grad_stride_h
        )
        dh += tl.load(grad_output_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
        output_offsets = (
            (t * batch_size + b_offs[:, None]) * output_feature_size
            + direction * hidden_size
            + h_offs[None, :]
        )
        h = tl.load(layer_output_ptr + output_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        dpre = dh * (1.0 - h * h)
        dpre_offsets = (t * batch_size + b_offs[:, None]) * hidden_size + h_offs[
            None, :
        ]
        tl.store(dpre_ptr + dpre_offsets, dpre, mask=mask)
        dh = tl.dot(dpre, weight_hh).to(tl.float32)
        dh = tl.where(mask, dh, 0.0)

    tl.store(grad_hx_ptr + grad_hidden_offsets, dh, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_bptt_vector_kernel(
    grad_output_ptr,
    grad_hidden_ptr,
    layer_output_ptr,
    weight_hh_ptr,
    dpre_ptr,
    grad_hx_ptr,
    dh_read_ptr,
    seq_len,
    batch_size,
    hidden_size,
    grad_stride_t,
    grad_stride_b,
    grad_stride_h,
    output_feature_size,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    state_index,
    direction: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    h_block_offs = tl.arange(0, BLOCK_H)
    chunk_offs = tl.arange(0, _CHUNK)
    num_h_blocks = tl.cdiv(hidden_size, BLOCK_H)
    state_base = (state_index * batch_size + batch_idx) * hidden_size
    read_base = batch_idx * hidden_size

    for h_block in range(num_h_blocks):
        h_offs = h_block * BLOCK_H + h_block_offs
        h_mask = h_offs < hidden_size
        dh = tl.load(grad_hidden_ptr + state_base + h_offs, mask=h_mask, other=0.0)
        tl.store(grad_hx_ptr + state_base + h_offs, dh, mask=h_mask)

    for step in range(seq_len):
        t = seq_len - 1 - step if direction == 0 else step
        for h_block in range(num_h_blocks):
            h_offs = h_block * BLOCK_H + h_block_offs
            h_mask = h_offs < hidden_size
            dh = tl.load(grad_hx_ptr + state_base + h_offs, mask=h_mask, other=0.0).to(
                tl.float32
            )
            go_offsets = (
                t * grad_stride_t
                + batch_idx * grad_stride_b
                + (direction * hidden_size + h_offs) * grad_stride_h
            )
            dh += tl.load(grad_output_ptr + go_offsets, mask=h_mask, other=0.0).to(
                tl.float32
            )
            output_offsets = (
                (t * batch_size + batch_idx) * output_feature_size
                + direction * hidden_size
                + h_offs
            )
            h = tl.load(layer_output_ptr + output_offsets, mask=h_mask, other=0.0).to(
                tl.float32
            )
            dpre = dh * (1.0 - h * h)
            dpre_offsets = (t * batch_size + batch_idx) * hidden_size + h_offs
            tl.store(dpre_ptr + dpre_offsets, dpre, mask=h_mask)

        # The full dpre vector must be visible before computing W_hh^T @ dpre.
        tl.debug_barrier()
        for h_block in range(num_h_blocks):
            j_offs = h_block * BLOCK_H + h_block_offs
            j_mask = j_offs < hidden_size
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for k_start in range(0, hidden_size, _CHUNK):
                k_offs = k_start + chunk_offs
                k_mask = k_offs < hidden_size
                dp = tl.load(
                    dpre_ptr + (t * batch_size + batch_idx) * hidden_size + k_offs,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                w_offsets = (
                    k_offs[:, None] * weight_hh_stride_h0
                    + j_offs[None, :] * weight_hh_stride_h1
                )
                w = tl.load(
                    weight_hh_ptr + w_offsets,
                    mask=k_mask[:, None] & j_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(dp[:, None] * w, axis=0)
            tl.store(dh_read_ptr + read_base + j_offs, acc, mask=j_mask)
        tl.debug_barrier()
        for h_block in range(num_h_blocks):
            h_offs = h_block * BLOCK_H + h_block_offs
            h_mask = h_offs < hidden_size
            dh = tl.load(dh_read_ptr + read_base + h_offs, mask=h_mask, other=0.0)
            tl.store(grad_hx_ptr + state_base + h_offs, dh, mask=h_mask)


@triton.jit
def rnn_tanh_bptt_step_kernel(
    grad_output_ptr,
    grad_state_ptr,
    layer_output_ptr,
    dpre_ptr,
    row_offset,
    active_batch,
    max_batch,
    hidden_size,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Ascend-friendly elementwise half of one BPTT time step."""
    batch_idx = tl.program_id(0)
    h_offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (batch_idx < active_batch) & (h_offs < hidden_size)
    state_offsets = (state_index * max_batch + batch_idx) * hidden_size + h_offs
    output_offsets = (
        (row_offset + batch_idx) * output_feature_size
        + direction * hidden_size
        + h_offs
    )
    dh = tl.load(grad_state_ptr + state_offsets, mask=mask, other=0.0).to(tl.float32)
    dh += tl.load(grad_output_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32)
    hidden = tl.load(layer_output_ptr + output_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    dpre = dh * (1.0 - hidden * hidden)
    dpre_offsets = (row_offset + batch_idx) * hidden_size + h_offs
    tl.store(dpre_ptr + dpre_offsets, dpre, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_bptt_matvec_kernel(
    dpre_ptr,
    weight_hh_ptr,
    grad_state_ptr,
    row_offset,
    active_batch,
    max_batch,
    hidden_size,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    state_index,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute dpre @ W_hh in a separate launch for Ascend synchronization."""
    batch_idx = tl.program_id(0)
    j_offs = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
    k_offs = tl.arange(0, BLOCK_K)
    j_mask = (batch_idx < active_batch) & (j_offs < hidden_size)
    acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    for k_start in range(0, hidden_size, BLOCK_K):
        ks = k_start + k_offs
        k_mask = (batch_idx < active_batch) & (ks < hidden_size)
        dpre = tl.load(
            dpre_ptr + (row_offset + batch_idx) * hidden_size + ks,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_hh_ptr
            + ks[:, None] * weight_hh_stride_h0
            + j_offs[None, :] * weight_hh_stride_h1,
            mask=k_mask[:, None] & j_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(dpre[:, None] * weight, axis=0)
    state_offsets = (state_index * max_batch + batch_idx) * hidden_size + j_offs
    tl.store(grad_state_ptr + state_offsets, acc, mask=j_mask)


@libentry()
@triton.jit
def rnn_tanh_dx_kernel(
    dpre_ptr,
    weight_ih_ptr,
    grad_input_ptr,
    rows,
    batch_size,
    input_size,
    hidden_size,
    grad_input_stride_t,
    grad_input_stride_b,
    grad_input_stride_i,
    weight_ih_stride_h,
    weight_ih_stride_i,
    dropout_p,
    dropout_seed,
    dropout_base,
    ADD: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    i_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for h_start in range(0, hidden_size, BLOCK_K):
        hs = h_start + h_offs
        dp = tl.load(
            dpre_ptr + row_offs[:, None] * hidden_size + hs[None, :],
            mask=(row_offs[:, None] < rows) & (hs[None, :] < hidden_size),
            other=0.0,
        )
        w = tl.load(
            weight_ih_ptr
            + hs[:, None] * weight_ih_stride_h
            + i_offs[None, :] * weight_ih_stride_i,
            mask=(hs[:, None] < hidden_size) & (i_offs[None, :] < input_size),
            other=0.0,
        )
        acc += tl.dot(dp, w)

    t = row_offs // batch_size
    b = row_offs - t * batch_size
    out_offsets = (
        t[:, None] * grad_input_stride_t
        + b[:, None] * grad_input_stride_b
        + i_offs[None, :] * grad_input_stride_i
    )
    mask = (row_offs[:, None] < rows) & (i_offs[None, :] < input_size)
    if APPLY_DROPOUT:
        random_offsets = dropout_base + row_offs[:, None] * input_size + i_offs[None, :]
        keep = tl.rand(dropout_seed, random_offsets) > dropout_p
        acc = tl.where(keep & mask, acc / (1.0 - dropout_p), 0.0)
    if ADD:
        acc += tl.load(grad_input_ptr + out_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    tl.store(grad_input_ptr + out_offsets, acc, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_dx_ascend_kernel(
    dpre_ptr,
    weight_ih_ptr,
    grad_input_ptr,
    rows,
    batch_size,
    input_size,
    hidden_size,
    grad_input_stride_t,
    grad_input_stride_b,
    grad_input_stride_i,
    weight_ih_stride_h,
    weight_ih_stride_i,
    dropout_p,
    dropout_seed,
    dropout_base,
    ADD: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Ascend dX path using scalar programs and a one-dimensional reduction."""
    row = tl.program_id(0)
    input_index = tl.program_id(1)
    h_offs = tl.arange(0, BLOCK_H)
    valid = (row < rows) & (input_index < input_size)
    acc = 0.0
    for h_start in range(0, hidden_size, BLOCK_H):
        hs = h_start + h_offs
        h_mask = valid & (hs < hidden_size)
        dpre = tl.load(
            dpre_ptr + row * hidden_size + hs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ih_ptr + hs * weight_ih_stride_h + input_index * weight_ih_stride_i,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(dpre * weight, axis=0)
    time_index = row // batch_size
    batch_index = row - time_index * batch_size
    output_offset = (
        time_index * grad_input_stride_t
        + batch_index * grad_input_stride_b
        + input_index * grad_input_stride_i
    )
    if APPLY_DROPOUT:
        random_offset = dropout_base + row * input_size + input_index
        keep = tl.rand(dropout_seed, random_offset) > dropout_p
        acc = tl.where(keep & valid, acc / (1.0 - dropout_p), 0.0)
    if ADD:
        acc += tl.load(grad_input_ptr + output_offset, mask=valid, other=0.0).to(
            tl.float32
        )
    tl.store(grad_input_ptr + output_offset, acc, mask=valid)


@libentry()
@triton.jit
def rnn_tanh_dw_ih_kernel(
    dpre_ptr,
    input_ptr,
    grad_weight_ptr,
    rows,
    batch_size,
    input_size,
    hidden_size,
    input_stride_t,
    input_stride_b,
    input_stride_i,
    dropout_p,
    dropout_seed,
    dropout_base,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    i_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for row_start in range(0, rows, BLOCK_K):
        rs = row_start + row_offs
        dp = tl.load(
            dpre_ptr + rs[None, :] * hidden_size + h_offs[:, None],
            mask=(rs[None, :] < rows) & (h_offs[:, None] < hidden_size),
            other=0.0,
        )
        t = rs // batch_size
        b = rs - t * batch_size
        input_offsets = (
            t[:, None] * input_stride_t
            + b[:, None] * input_stride_b
            + i_offs[None, :] * input_stride_i
        )
        x_mask = (rs[:, None] < rows) & (i_offs[None, :] < input_size)
        x = tl.load(input_ptr + input_offsets, mask=x_mask, other=0.0)
        if APPLY_DROPOUT:
            random_offsets = dropout_base + rs[:, None] * input_size + i_offs[None, :]
            keep = tl.rand(dropout_seed, random_offsets) > dropout_p
            x = tl.where(keep & x_mask, x / (1.0 - dropout_p), 0.0)
        acc += tl.dot(dp, x)
    mask = (h_offs[:, None] < hidden_size) & (i_offs[None, :] < input_size)
    grad_offsets = h_offs[:, None] * input_size + i_offs[None, :]
    tl.store(grad_weight_ptr + grad_offsets, acc, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_dw_hh_kernel(
    dpre_ptr,
    layer_output_ptr,
    hx_ptr,
    grad_weight_ptr,
    rows,
    seq_len,
    batch_size,
    hidden_size,
    output_feature_size,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    state_index,
    direction: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h0_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    h1_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for row_start in range(0, rows, BLOCK_K):
        rs = row_start + row_offs
        dp = tl.load(
            dpre_ptr + rs[None, :] * hidden_size + h0_offs[:, None],
            mask=(rs[None, :] < rows) & (h0_offs[:, None] < hidden_size),
            other=0.0,
        )
        t = rs // batch_size
        b = rs - t * batch_size
        uses_hx = t == 0 if direction == 0 else t == seq_len - 1
        previous_t = t - 1 if direction == 0 else t + 1
        layer_offsets = (
            (previous_t[:, None] * batch_size + b[:, None]) * output_feature_size
            + direction * hidden_size
            + h1_offs[None, :]
        )
        layer_mask = (
            (rs[:, None] < rows) & (h1_offs[None, :] < hidden_size) & ~uses_hx[:, None]
        )
        previous = tl.load(layer_output_ptr + layer_offsets, mask=layer_mask, other=0.0)
        hx_offsets = (
            state_index * hx_stride_state
            + b[:, None] * hx_stride_b
            + h1_offs[None, :] * hx_stride_h
        )
        hx_value = tl.load(
            hx_ptr + hx_offsets,
            mask=(rs[:, None] < rows) & (h1_offs[None, :] < hidden_size),
            other=0.0,
        )
        previous = tl.where(uses_hx[:, None], hx_value, previous)
        acc += tl.dot(dp, previous)
    mask = (h0_offs[:, None] < hidden_size) & (h1_offs[None, :] < hidden_size)
    tl.store(
        grad_weight_ptr + h0_offs[:, None] * hidden_size + h1_offs[None, :],
        acc,
        mask=mask,
    )


@libentry()
@triton.jit
def rnn_tanh_dw_hh_ascend_kernel(
    dpre_ptr,
    layer_output_ptr,
    hx_ptr,
    grad_weight_ptr,
    rows,
    seq_len,
    batch_size,
    hidden_size,
    output_feature_size,
    hx_stride_state,
    hx_stride_b,
    hx_stride_h,
    state_index,
    direction: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ascend dW_hh path avoiding a zero-stride conditional dot operand."""
    h0 = tl.program_id(0)
    h1 = tl.program_id(1)
    row_offs = tl.arange(0, BLOCK_K)
    acc = 0.0
    for row_start in range(0, rows, BLOCK_K):
        rs = row_start + row_offs
        row_mask = rs < rows
        dpre = tl.load(
            dpre_ptr + rs * hidden_size + h0,
            mask=row_mask & (h0 < hidden_size),
            other=0.0,
        ).to(tl.float32)
        time_index = rs // batch_size
        batch_index = rs - time_index * batch_size
        uses_hx = time_index == 0 if direction == 0 else time_index == seq_len - 1
        previous_time = time_index - 1 if direction == 0 else time_index + 1
        previous_offsets = (
            (previous_time * batch_size + batch_index) * output_feature_size
            + direction * hidden_size
            + h1
        )
        previous_mask = row_mask & (h1 < hidden_size) & ~uses_hx
        previous = tl.load(
            layer_output_ptr + previous_offsets,
            mask=previous_mask,
            other=0.0,
        ).to(tl.float32)
        hx_offsets = (
            state_index * hx_stride_state + batch_index * hx_stride_b + h1 * hx_stride_h
        )
        initial = tl.load(
            hx_ptr + hx_offsets,
            mask=row_mask & (h1 < hidden_size),
            other=0.0,
        ).to(tl.float32)
        previous = tl.where(uses_hx, initial, previous)
        acc += tl.sum(dpre * previous, axis=0)
    mask = (h0 < hidden_size) & (h1 < hidden_size)
    tl.store(
        grad_weight_ptr + h0 * hidden_size + h1,
        acc,
        mask=mask,
    )


@libentry()
@triton.jit
def rnn_tanh_dbias_kernel(
    dpre_ptr,
    grad_bias_ih_ptr,
    grad_bias_hh_ptr,
    rows,
    hidden_size,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    h_offs = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    r_offs = tl.arange(0, BLOCK_R)
    # Accumulate row tiles before reducing. Older Triton backends can assert
    # when a loop containing a reduction feeds both independent bias stores.
    # This also performs the cross-row reduction only once per program.
    partial = tl.zeros([BLOCK_R, BLOCK_H], dtype=tl.float32)
    for r_start in range(0, rows, BLOCK_R):
        rs = r_start + r_offs
        values = tl.load(
            dpre_ptr + rs[:, None] * hidden_size + h_offs[None, :],
            mask=(rs[:, None] < rows) & (h_offs[None, :] < hidden_size),
            other=0.0,
        ).to(tl.float32)
        partial += values
    acc = tl.sum(partial, axis=0)
    mask = h_offs < hidden_size
    tl.store(grad_bias_ih_ptr + h_offs, acc, mask=mask)
    tl.store(grad_bias_hh_ptr + h_offs, acc, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_packed_forward_step_dot_kernel(
    input_ptr,
    hidden_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    previous_ptr,
    row_offset,
    active_batch,
    max_batch,
    input_size,
    hidden_size,
    input_stride_row,
    input_stride_i,
    weight_ih_stride_h,
    weight_ih_stride_i,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    bias_ih_stride,
    bias_hh_stride,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    dropout_p,
    dropout_seed,
    dropout_base,
    HAS_BIAS: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_start = tl.program_id(0) * BLOCK_B
    b_offs = batch_start + tl.arange(0, BLOCK_B)
    i_offs = tl.arange(0, BLOCK_I)
    h_offs = tl.arange(0, BLOCK_H)
    b_mask = b_offs < active_batch
    h_mask = h_offs < hidden_size
    hidden_offsets = (state_index * max_batch + b_offs[:, None]) * hidden_size + h_offs[
        None, :
    ]
    hidden = tl.load(
        hidden_ptr + hidden_offsets,
        mask=b_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    previous_offsets = (row_offset + b_offs[:, None]) * hidden_size + h_offs[None, :]
    tl.store(
        previous_ptr + previous_offsets,
        hidden,
        mask=b_mask[:, None] & h_mask[None, :],
    )

    input_offsets = (row_offset + b_offs[:, None]) * input_stride_row + i_offs[
        None, :
    ] * input_stride_i
    input_mask = b_mask[:, None] & (i_offs[None, :] < input_size)
    x = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    if APPLY_DROPOUT:
        random_offsets = (
            dropout_base + (row_offset + b_offs[:, None]) * input_size + i_offs[None, :]
        )
        keep = tl.rand(dropout_seed, random_offsets) > dropout_p
        x = tl.where(keep & input_mask, x / (1.0 - dropout_p), 0.0)

    w_ih = tl.load(
        weight_ih_ptr
        + h_offs[:, None] * weight_ih_stride_h
        + i_offs[None, :] * weight_ih_stride_i,
        mask=(h_offs[:, None] < hidden_size) & (i_offs[None, :] < input_size),
        other=0.0,
    )
    w_hh = tl.load(
        weight_hh_ptr
        + h_offs[:, None] * weight_hh_stride_h0
        + h_offs[None, :] * weight_hh_stride_h1,
        mask=h_mask[:, None] & h_mask[None, :],
        other=0.0,
    )
    acc = tl.dot(x.to(w_ih.dtype), tl.trans(w_ih), input_precision=_DOT_PRECISION).to(
        tl.float32
    )
    acc += tl.dot(
        hidden.to(w_hh.dtype), tl.trans(w_hh), input_precision=_DOT_PRECISION
    ).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(
            bias_ih_ptr + h_offs * bias_ih_stride, mask=h_mask, other=0.0
        ).to(tl.float32)
        bias += tl.load(
            bias_hh_ptr + h_offs * bias_hh_stride, mask=h_mask, other=0.0
        ).to(tl.float32)
        acc += bias[None, :]
    current = tl_extra_shim.tanh(acc)
    current = tl.where(b_mask[:, None] & h_mask[None, :], current, 0.0)
    tl.store(
        hidden_ptr + hidden_offsets,
        current,
        mask=b_mask[:, None] & h_mask[None, :],
    )
    output_offsets = (
        (row_offset + b_offs[:, None]) * output_feature_size
        + direction * hidden_size
        + h_offs[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        current,
        mask=b_mask[:, None] & h_mask[None, :],
    )


@libentry()
@triton.jit
def rnn_tanh_packed_forward_step_vector_kernel(
    input_ptr,
    hidden_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    previous_ptr,
    row_offset,
    active_batch,
    max_batch,
    input_size,
    hidden_size,
    input_stride_row,
    input_stride_i,
    weight_ih_stride_h,
    weight_ih_stride_i,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    bias_ih_stride,
    bias_hh_stride,
    output_feature_size,
    state_index,
    direction: tl.constexpr,
    dropout_p,
    dropout_seed,
    dropout_base,
    HAS_BIAS: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx >= active_batch:
        return
    h_block_offs = tl.arange(0, BLOCK_H)
    chunk_offs = tl.arange(0, _CHUNK)
    num_h_blocks = tl.cdiv(hidden_size, BLOCK_H)
    hidden_base = (state_index * max_batch + batch_idx) * hidden_size
    previous_base = (row_offset + batch_idx) * hidden_size
    for h_block in range(num_h_blocks):
        h_offs = h_block * BLOCK_H + h_block_offs
        h_mask = h_offs < hidden_size
        previous = tl.load(hidden_ptr + hidden_base + h_offs, mask=h_mask, other=0.0)
        tl.store(previous_ptr + previous_base + h_offs, previous, mask=h_mask)
    for h_block in range(num_h_blocks):
        h_offs = h_block * BLOCK_H + h_block_offs
        h_mask = h_offs < hidden_size
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i_start in range(0, input_size, _CHUNK):
            i_offs = i_start + chunk_offs
            i_mask = i_offs < input_size
            x = tl.load(
                input_ptr
                + (row_offset + batch_idx) * input_stride_row
                + i_offs * input_stride_i,
                mask=i_mask,
                other=0.0,
            ).to(tl.float32)
            if APPLY_DROPOUT:
                random_offsets = (
                    dropout_base + (row_offset + batch_idx) * input_size + i_offs
                )
                keep = tl.rand(dropout_seed, random_offsets) > dropout_p
                x = tl.where(keep & i_mask, x / (1.0 - dropout_p), 0.0)
            w = tl.load(
                weight_ih_ptr
                + h_offs[:, None] * weight_ih_stride_h
                + i_offs[None, :] * weight_ih_stride_i,
                mask=h_mask[:, None] & i_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        for j_start in range(0, hidden_size, _CHUNK):
            j_offs = j_start + chunk_offs
            j_mask = j_offs < hidden_size
            previous = tl.load(
                previous_ptr + previous_base + j_offs, mask=j_mask, other=0.0
            ).to(tl.float32)
            w = tl.load(
                weight_hh_ptr
                + h_offs[:, None] * weight_hh_stride_h0
                + j_offs[None, :] * weight_hh_stride_h1,
                mask=h_mask[:, None] & j_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * previous[None, :], axis=1)
        if HAS_BIAS:
            acc += tl.load(
                bias_ih_ptr + h_offs * bias_ih_stride,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.load(
                bias_hh_ptr + h_offs * bias_hh_stride,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
        current = tl_extra_shim.tanh(acc)
        tl.store(hidden_ptr + hidden_base + h_offs, current, mask=h_mask)
        output_offsets = (
            (row_offset + batch_idx) * output_feature_size
            + direction * hidden_size
            + h_offs
        )
        tl.store(output_ptr + output_offsets, current, mask=h_mask)


@libentry()
@triton.jit
def rnn_tanh_packed_bptt_step_dot_kernel(
    grad_output_ptr,
    grad_hidden_work_ptr,
    layer_output_ptr,
    weight_hh_ptr,
    dpre_ptr,
    row_offset,
    active_batch,
    max_batch,
    hidden_size,
    output_feature_size,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    state_index,
    direction: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_start = tl.program_id(0) * BLOCK_B
    b_offs = batch_start + tl.arange(0, BLOCK_B)
    h_offs = tl.arange(0, BLOCK_H)
    mask = (b_offs[:, None] < active_batch) & (h_offs[None, :] < hidden_size)
    state_offsets = (state_index * max_batch + b_offs[:, None]) * hidden_size + h_offs[
        None, :
    ]
    dh = tl.load(grad_hidden_work_ptr + state_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    output_offsets = (
        (row_offset + b_offs[:, None]) * output_feature_size
        + direction * hidden_size
        + h_offs[None, :]
    )
    dh += tl.load(grad_output_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(layer_output_ptr + output_offsets, mask=mask, other=0.0).to(tl.float32)
    dpre = dh * (1.0 - h * h)
    dpre_offsets = (row_offset + b_offs[:, None]) * hidden_size + h_offs[None, :]
    tl.store(dpre_ptr + dpre_offsets, dpre, mask=mask)
    weight_hh = tl.load(
        weight_hh_ptr
        + h_offs[:, None] * weight_hh_stride_h0
        + h_offs[None, :] * weight_hh_stride_h1,
        mask=(h_offs[:, None] < hidden_size) & (h_offs[None, :] < hidden_size),
        other=0.0,
    )
    dh_previous = tl.dot(dpre, weight_hh).to(tl.float32)
    tl.store(grad_hidden_work_ptr + state_offsets, dh_previous, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_packed_bptt_step_vector_kernel(
    grad_output_ptr,
    grad_hidden_work_ptr,
    layer_output_ptr,
    weight_hh_ptr,
    dpre_ptr,
    dh_read_ptr,
    row_offset,
    active_batch,
    max_batch,
    hidden_size,
    output_feature_size,
    weight_hh_stride_h0,
    weight_hh_stride_h1,
    state_index,
    direction: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx >= active_batch:
        return
    h_block_offs = tl.arange(0, BLOCK_H)
    chunk_offs = tl.arange(0, _CHUNK)
    num_h_blocks = tl.cdiv(hidden_size, BLOCK_H)
    state_base = (state_index * max_batch + batch_idx) * hidden_size
    dpre_base = (row_offset + batch_idx) * hidden_size
    for h_block in range(num_h_blocks):
        h_offs = h_block * BLOCK_H + h_block_offs
        h_mask = h_offs < hidden_size
        output_offsets = (
            (row_offset + batch_idx) * output_feature_size
            + direction * hidden_size
            + h_offs
        )
        dh = tl.load(
            grad_hidden_work_ptr + state_base + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        dh += tl.load(grad_output_ptr + output_offsets, mask=h_mask, other=0.0).to(
            tl.float32
        )
        h = tl.load(layer_output_ptr + output_offsets, mask=h_mask, other=0.0).to(
            tl.float32
        )
        dpre = dh * (1.0 - h * h)
        tl.store(dpre_ptr + dpre_base + h_offs, dpre, mask=h_mask)
    tl.debug_barrier()
    for h_block in range(num_h_blocks):
        j_offs = h_block * BLOCK_H + h_block_offs
        j_mask = j_offs < hidden_size
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k_start in range(0, hidden_size, _CHUNK):
            k_offs = k_start + chunk_offs
            k_mask = k_offs < hidden_size
            dp = tl.load(dpre_ptr + dpre_base + k_offs, mask=k_mask, other=0.0).to(
                tl.float32
            )
            w = tl.load(
                weight_hh_ptr
                + k_offs[:, None] * weight_hh_stride_h0
                + j_offs[None, :] * weight_hh_stride_h1,
                mask=k_mask[:, None] & j_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(dp[:, None] * w, axis=0)
        tl.store(dh_read_ptr + batch_idx * hidden_size + j_offs, acc, mask=j_mask)
    tl.debug_barrier()
    for h_block in range(num_h_blocks):
        h_offs = h_block * BLOCK_H + h_block_offs
        h_mask = h_offs < hidden_size
        dh = tl.load(
            dh_read_ptr + batch_idx * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        )
        tl.store(grad_hidden_work_ptr + state_base + h_offs, dh, mask=h_mask)


@libentry()
@triton.jit
def rnn_tanh_packed_dx_kernel(
    dpre_ptr,
    weight_ptr,
    grad_input_ptr,
    rows,
    input_size,
    hidden_size,
    weight_stride_h,
    weight_stride_i,
    dropout_p,
    dropout_seed,
    dropout_base,
    ADD: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    i_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for h_start in range(0, hidden_size, BLOCK_K):
        hs = h_start + h_offs
        dp = tl.load(
            dpre_ptr + row_offs[:, None] * hidden_size + hs[None, :],
            mask=(row_offs[:, None] < rows) & (hs[None, :] < hidden_size),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + hs[:, None] * weight_stride_h
            + i_offs[None, :] * weight_stride_i,
            mask=(hs[:, None] < hidden_size) & (i_offs[None, :] < input_size),
            other=0.0,
        )
        acc += tl.dot(dp, weight)
    offsets = row_offs[:, None] * input_size + i_offs[None, :]
    mask = (row_offs[:, None] < rows) & (i_offs[None, :] < input_size)
    if APPLY_DROPOUT:
        random_offsets = dropout_base + offsets
        keep = tl.rand(dropout_seed, random_offsets) > dropout_p
        acc = tl.where(keep & mask, acc / (1.0 - dropout_p), 0.0)
    if ADD:
        acc += tl.load(grad_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_input_ptr + offsets, acc, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_packed_dw_kernel(
    dpre_ptr,
    input_ptr,
    grad_weight_ptr,
    rows,
    input_size,
    hidden_size,
    input_stride_row,
    input_stride_i,
    dropout_p,
    dropout_seed,
    dropout_base,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    i_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for row_start in range(0, rows, BLOCK_K):
        rs = row_start + row_offs
        dp = tl.load(
            dpre_ptr + rs[None, :] * hidden_size + h_offs[:, None],
            mask=(rs[None, :] < rows) & (h_offs[:, None] < hidden_size),
            other=0.0,
        )
        input_offsets = (
            rs[:, None] * input_stride_row + i_offs[None, :] * input_stride_i
        )
        input_mask = (rs[:, None] < rows) & (i_offs[None, :] < input_size)
        x = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        if APPLY_DROPOUT:
            random_offsets = dropout_base + rs[:, None] * input_size + i_offs[None, :]
            keep = tl.rand(dropout_seed, random_offsets) > dropout_p
            x = tl.where(keep & input_mask, x / (1.0 - dropout_p), 0.0)
        acc += tl.dot(dp, x)
    mask = (h_offs[:, None] < hidden_size) & (i_offs[None, :] < input_size)
    offsets = h_offs[:, None] * input_size + i_offs[None, :]
    tl.store(grad_weight_ptr + offsets, acc, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_batch_first_copy_kernel(
    source_ptr,
    destination_ptr,
    seq_len,
    batch_size,
    feature_size,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = seq_len * batch_size * feature_size
    mask = offsets < total
    feature = offsets % feature_size
    tmp = offsets // feature_size
    t = tmp // batch_size
    b = tmp - t * batch_size
    destination_offsets = (b * seq_len + t) * feature_size + feature
    value = tl.load(source_ptr + offsets, mask=mask, other=0.0)
    tl.store(destination_ptr + destination_offsets, value, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_copy_3d_kernel(
    source_ptr,
    destination_ptr,
    size0,
    size1,
    size2,
    source_stride0,
    source_stride1,
    source_stride2,
    BLOCK: tl.constexpr,
):
    """Materialize an arbitrary 3-D view without calling a Torch copy op."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = size0 * size1 * size2
    mask = offsets < total
    index2 = offsets % size2
    tmp = offsets // size2
    index1 = tmp % size1
    index0 = tmp // size1
    source_offsets = (
        index0 * source_stride0 + index1 * source_stride1 + index2 * source_stride2
    )
    value = tl.load(source_ptr + source_offsets, mask=mask, other=0.0)
    tl.store(destination_ptr + offsets, value, mask=mask)


@libentry()
@triton.jit
def rnn_tanh_add_kernel(destination_ptr, source_ptr, total, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    destination = tl.load(destination_ptr + offsets, mask=mask, other=0.0)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0)
    tl.store(destination_ptr + offsets, destination + source, mask=mask)


def _empty(shape, reference, dtype=None):
    # Allocation is the only PyTorch tensor operation used by the launcher.
    return torch.empty(
        shape,
        dtype=reference.dtype if dtype is None else dtype,
        device=reference.device,
    )


def _validate(input, hx, params, has_biases, num_layers, dropout, bidirectional):
    if input.ndim != 3:
        raise RuntimeError(f"rnn_tanh: expected a 3-D input, got {input.ndim}-D")
    if hx.ndim != 3:
        raise RuntimeError(f"rnn_tanh: expected a 3-D hidden state, got {hx.ndim}-D")
    if num_layers <= 0:
        raise RuntimeError("rnn_tanh: num_layers must be greater than zero")
    if dropout < 0.0 or dropout > 1.0:
        raise RuntimeError("rnn_tanh: dropout must be in the range [0, 1]")
    directions = 2 if bidirectional else 1
    params_per_state = 4 if has_biases else 2
    expected_params = num_layers * directions * params_per_state
    if len(params) != expected_params:
        raise RuntimeError(
            f"rnn_tanh: expected {expected_params} parameters, got {len(params)}"
        )
    if hx.shape[0] != num_layers * directions:
        raise RuntimeError(
            "rnn_tanh: hidden state has an invalid first dimension: "
            f"expected {num_layers * directions}, got {hx.shape[0]}"
        )


def _unpack_state_params(params, state_index, has_biases):
    width = 4 if has_biases else 2
    base = state_index * width
    if has_biases:
        return params[base], params[base + 1], params[base + 2], params[base + 3]
    return params[base], params[base + 1], None, None


def _dropout_rng(random_values, enabled):
    if not enabled:
        return 0, 0
    # tl.rand consumes one Philox counter for every offset supplied by these
    # kernels, so reserve the complete, non-overlapping counter range.
    return philox_backend_seed_offset(random_values)


def _launch_forward(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
    dropout_seed,
    dropout_offset,
    prefer_persistent_dot,
):
    ascend_stream = None
    if runtime.device.vendor_name in ("ascend", "metax"):
        # Resolve the caller's stream once per invocation, not once for every
        # recurrent launch. Never retain it across calls or stream contexts.
        driver = triton.runtime.driver.active
        ascend_stream = driver.get_current_stream(driver.get_current_device())
    if batch_first:
        batch_size, seq_len, first_input_size = input.shape
        input_strides = (input.stride(1), input.stride(0), input.stride(2))
    else:
        seq_len, batch_size, first_input_size = input.shape
        input_strides = input.stride()
    directions = 2 if bidirectional else 1
    hidden_size = hx.shape[2]
    hidden = _empty((num_layers * directions, batch_size, hidden_size), input)
    hidden_read = None
    layer_outputs = []
    current_input = input
    current_input_size = first_input_size
    current_strides = input_strides

    for layer in range(num_layers):
        feature_size = directions * hidden_size
        layer_output = _empty((seq_len, batch_size, feature_size), input)
        apply_dropout = train and dropout > 0.0 and layer > 0
        dropout_base = (
            dropout_offset + (layer - 1) * seq_len * batch_size * current_input_size
        )
        for direction in range(directions):
            state_index = layer * directions + direction
            weight_ih, weight_hh, bias_ih, bias_hh = _unpack_state_params(
                params, state_index, has_biases
            )
            matrix_shape = (
                current_input_size <= 256
                and hidden_size <= 256
                and current_input_size >= 16
                and hidden_size >= 16
            )
            vendor = runtime.device.vendor_name
            use_split_persistent = (
                matrix_shape
                and hidden_size <= 128
                and (
                    (
                        vendor == "nvidia"
                        and input.dtype != torch.bfloat16
                        and hidden_size >= 128
                    )
                )
            )
            use_dot = (
                matrix_shape
                # NVIDIA's persistent dot kernel supports bf16 tensor cores,
                # while other backends retain their established selector.
                and (input.dtype != torch.bfloat16 or prefer_persistent_dot)
                # The vector kernel is faster for small NVIDIA states; larger
                # states benefit from the tensor-core dot path.
                and (not prefer_persistent_dot or hidden_size > 64)
            )
            use_ascend_chunked = (
                vendor == "ascend"
                and matrix_shape
                and input.dtype == torch.bfloat16
                and hidden_size == 128
            )
            use_ascend_tiled = (
                vendor == "ascend"
                and matrix_shape
                and not use_split_persistent
                and not use_ascend_chunked
            ) or (
                vendor == "metax"
                and matrix_shape
                and input.dtype != torch.bfloat16
                and hidden_size >= 128
            )
            if use_split_persistent:
                rows = seq_len * batch_size
                input_linear = _empty((seq_len, batch_size, hidden_size), input)
                block_m = 16
                block_n = min(64, max(16, triton.next_power_of_2(hidden_size)))
                block_k = min(64, max(16, triton.next_power_of_2(current_input_size)))
                linear_grid = (
                    triton.cdiv(rows, block_m),
                    triton.cdiv(hidden_size, block_n),
                )
                launch_warps = 1 if vendor == "ascend" else 4
                launch_stages = 1 if vendor in ("ascend", "metax") else 2
                rnn_tanh_input_linear_kernel[linear_grid](
                    current_input,
                    weight_ih,
                    bias_ih,
                    bias_hh,
                    input_linear,
                    rows,
                    batch_size,
                    current_input_size,
                    hidden_size,
                    *current_strides,
                    *weight_ih.stride(),
                    bias_ih.stride(0) if has_biases else 0,
                    bias_hh.stride(0) if has_biases else 0,
                    dropout,
                    dropout_seed,
                    dropout_base,
                    HAS_BIAS=has_biases,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=launch_warps,
                    num_stages=launch_stages,
                )
                block_b = 16
                block_h = max(16, triton.next_power_of_2(hidden_size))
                recurrent_grid = (triton.cdiv(batch_size, block_b),)
                rnn_tanh_recurrent_persistent_kernel[recurrent_grid](
                    hx,
                    weight_hh,
                    input_linear,
                    layer_output,
                    hidden,
                    seq_len,
                    batch_size,
                    hidden_size,
                    *hx.stride(),
                    *weight_hh.stride(),
                    feature_size,
                    state_index,
                    direction,
                    BLOCK_B=block_b,
                    BLOCK_H=block_h,
                    num_warps=(
                        1 if vendor == "ascend" else (8 if block_h >= 128 else 4)
                    ),
                    num_stages=launch_stages,
                )
            elif use_ascend_chunked:
                rows = seq_len * batch_size
                input_linear = _empty((seq_len, batch_size, hidden_size), input)
                block_m = 16
                block_n = 32 if hidden_size <= 32 else 64
                block_k = 32 if max(current_input_size, hidden_size) <= 32 else 64
                linear_grid = (
                    triton.cdiv(rows, block_m),
                    triton.cdiv(hidden_size, block_n),
                )
                rnn_tanh_input_linear_kernel[linear_grid](
                    current_input,
                    weight_ih,
                    bias_ih,
                    bias_hh,
                    input_linear,
                    rows,
                    batch_size,
                    current_input_size,
                    hidden_size,
                    *current_strides,
                    *weight_ih.stride(),
                    bias_ih.stride(0) if has_biases else 0,
                    bias_hh.stride(0) if has_biases else 0,
                    dropout,
                    dropout_seed,
                    dropout_base,
                    HAS_BIAS=has_biases,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=1,
                    num_stages=1,
                )
                chunk_size = 3
                block_b = 8
                block_h = max(16, triton.next_power_of_2(hidden_size))
                recurrent_grid = (triton.cdiv(batch_size, block_b),)
                direct_grid = (*recurrent_grid, 1, 1)
                compiled_chunk = None
                for chunk_start in range(0, seq_len, chunk_size):
                    steps = min(chunk_size, seq_len - chunk_start)
                    final_chunk = chunk_start + steps == seq_len
                    if compiled_chunk is not None and not final_chunk:
                        compiled_chunk(
                            hx,
                            weight_hh,
                            input_linear,
                            layer_output,
                            hidden,
                            chunk_start,
                            *hx.stride(),
                            *weight_hh.stride(),
                            feature_size,
                            state_index,
                            stream=ascend_stream,
                        )
                    else:
                        compiled_kernel, _ = rnn_tanh_recurrent_chunk_ascend_kernel[
                            recurrent_grid
                        ](
                            hx,
                            weight_hh,
                            input_linear,
                            layer_output,
                            hidden,
                            chunk_start,
                            *hx.stride(),
                            *weight_hh.stride(),
                            feature_size,
                            state_index,
                            seq_len,
                            batch_size,
                            hidden_size,
                            direction,
                            FIRST_CHUNK=chunk_start == 0,
                            FINAL_CHUNK=final_chunk,
                            STEPS=steps,
                            BLOCK_B=block_b,
                            BLOCK_H=block_h,
                            num_warps=1,
                            num_stages=1,
                        )
                        # Reuse only a full, non-initial, non-final chunk.
                        # Its dynamic start index is neither scalar-specialized
                        # 0 nor 1, and subsequent starts share its alignment.
                        if chunk_start > 1 and not final_chunk:
                            compiled_chunk = compiled_kernel[direct_grid]
            elif use_ascend_tiled:
                use_ascend_composed_tanh = hidden_size <= 128
                if not use_ascend_composed_tanh and hidden_read is None:
                    hidden_read = _empty((batch_size, hidden_size), input)
                rows = seq_len * batch_size
                input_linear = _empty((seq_len, batch_size, hidden_size), input)
                block_m = 16
                block_n = 32 if hidden_size <= 32 else 64
                block_k = 32 if max(current_input_size, hidden_size) <= 32 else 64
                linear_grid = (
                    triton.cdiv(rows, block_m),
                    triton.cdiv(hidden_size, block_n),
                )
                rnn_tanh_input_linear_kernel[linear_grid](
                    current_input,
                    weight_ih,
                    bias_ih,
                    bias_hh,
                    input_linear,
                    rows,
                    batch_size,
                    current_input_size,
                    hidden_size,
                    *current_strides,
                    *weight_ih.stride(),
                    bias_ih.stride(0) if has_biases else 0,
                    bias_hh.stride(0) if has_biases else 0,
                    dropout,
                    dropout_seed,
                    dropout_base,
                    HAS_BIAS=has_biases,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=1,
                    num_stages=1,
                )
                # The 128-wide recurrent state fits in one reduction tile;
                # avoid two matrix iterations and their intermediate transfers.
                if hidden_size == 128:
                    block_k = 128
                recurrent_grid = (
                    triton.cdiv(batch_size, block_m),
                    triton.cdiv(hidden_size, block_n),
                )
                compiled_middle = None
                reusable_step = 2 if direction == 0 else 1
                direct_grid = (*recurrent_grid, 1)
                if use_ascend_composed_tanh:
                    activation_block_n = hidden_size
                    activation_block_m = min(
                        batch_size, max(1, 512 // activation_block_n)
                    )
                    activation_grid = (
                        triton.cdiv(batch_size, activation_block_m),
                        1,
                    )
                    activation_direct_grid = (*activation_grid, 1)
                    compiled_activation = None
                    for step in range(seq_len):
                        time_index = step if direction == 0 else seq_len - 1 - step
                        if compiled_middle is not None:
                            compiled_middle(
                                hx,
                                weight_hh,
                                input_linear,
                                layer_output,
                                time_index,
                                *hx.stride(),
                                *weight_hh.stride(),
                                feature_size,
                                state_index,
                                stream=ascend_stream,
                            )
                        else:
                            compiled_kernel, _ = rnn_tanh_recurrent_addmm_ascend_kernel[
                                recurrent_grid
                            ](
                                hx,
                                weight_hh,
                                input_linear,
                                layer_output,
                                time_index,
                                batch_size,
                                hidden_size,
                                *hx.stride(),
                                *weight_hh.stride(),
                                feature_size,
                                state_index,
                                direction,
                                FIRST_STEP=step == 0,
                                BLOCK_M=block_m,
                                BLOCK_N=block_n,
                                BLOCK_K=block_k,
                                num_warps=1,
                                num_stages=1,
                            )
                            if step == reusable_step and time_index > 1:
                                compiled_middle = compiled_kernel[direct_grid]
                        final_step = step == seq_len - 1
                        if compiled_activation is not None and not final_step:
                            compiled_activation(
                                layer_output,
                                hidden,
                                time_index,
                                feature_size,
                                state_index,
                                stream=ascend_stream,
                            )
                        else:
                            compiled_kernel, _ = rnn_tanh_activation_ascend_kernel[
                                activation_grid
                            ](
                                layer_output,
                                hidden,
                                time_index,
                                batch_size,
                                hidden_size,
                                feature_size,
                                state_index,
                                direction,
                                FINAL_STEP=final_step,
                                BLOCK_M=activation_block_m,
                                BLOCK_N=activation_block_n,
                                num_warps=1,
                                num_stages=1,
                            )
                            if (
                                step == reusable_step
                                and time_index > 1
                                and not final_step
                            ):
                                compiled_activation = compiled_kernel[
                                    activation_direct_grid
                                ]
                else:
                    for step in range(seq_len):
                        time_index = step if direction == 0 else seq_len - 1 - step
                        # LibEntry rebuilds the launch key for every recurrent step.
                        # Capture a non-initial launch whose time index is not the
                        # scalar-specialized 0/1, then reuse that FlagGems kernel.
                        if compiled_middle is not None and step < seq_len - 1:
                            compiled_middle(
                                hx,
                                weight_hh,
                                bias_hh,
                                input_linear,
                                hidden_read,
                                layer_output,
                                hidden,
                                time_index,
                                *hx.stride(),
                                *weight_hh.stride(),
                                bias_hh.stride(0) if has_biases else 0,
                                feature_size,
                                state_index,
                                stream=ascend_stream,
                            )
                        else:
                            compiled_kernel, _ = rnn_tanh_recurrent_ascend_kernel[
                                recurrent_grid
                            ](
                                hx,
                                weight_hh,
                                bias_hh,
                                input_linear,
                                hidden_read,
                                layer_output,
                                hidden,
                                time_index,
                                batch_size,
                                hidden_size,
                                *hx.stride(),
                                *weight_hh.stride(),
                                bias_hh.stride(0) if has_biases else 0,
                                feature_size,
                                state_index,
                                direction,
                                FIRST_STEP=step == 0,
                                FINAL_STEP=step == seq_len - 1,
                                HAS_BIAS=has_biases,
                                BLOCK_M=block_m,
                                BLOCK_N=block_n,
                                BLOCK_K=block_k,
                                num_warps=1,
                                num_stages=1,
                            )
                            if (
                                step == reusable_step
                                and time_index > 1
                                and step < seq_len - 1
                            ):
                                compiled_middle = compiled_kernel[direct_grid]
            elif use_dot:
                block_b = min(16, triton.next_power_of_2(batch_size))
                block_b = max(block_b, 16)
                block_i = max(16, triton.next_power_of_2(current_input_size))
                block_h = max(16, triton.next_power_of_2(hidden_size))
                grid = (triton.cdiv(batch_size, block_b),)
                rnn_tanh_forward_dot_kernel[grid](
                    current_input,
                    hx,
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    layer_output,
                    hidden,
                    seq_len,
                    batch_size,
                    current_input_size,
                    hidden_size,
                    *current_strides,
                    *hx.stride(),
                    *weight_ih.stride(),
                    *weight_hh.stride(),
                    bias_ih.stride(0) if has_biases else 0,
                    bias_hh.stride(0) if has_biases else 0,
                    feature_size,
                    state_index,
                    direction,
                    dropout,
                    dropout_seed,
                    dropout_base,
                    HAS_BIAS=has_biases,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_B=block_b,
                    BLOCK_I=block_i,
                    BLOCK_H=block_h,
                    num_warps=8 if block_h >= 128 else 4,
                    num_stages=2,
                )
            else:
                if hidden_read is None:
                    hidden_read = _empty((batch_size, hidden_size), input)
                grid = (batch_size,)
                rnn_tanh_forward_vector_kernel[grid](
                    current_input,
                    hx,
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    layer_output,
                    hidden,
                    hidden_read,
                    seq_len,
                    batch_size,
                    current_input_size,
                    hidden_size,
                    *current_strides,
                    *hx.stride(),
                    *weight_ih.stride(),
                    *weight_hh.stride(),
                    bias_ih.stride(0) if has_biases else 0,
                    bias_hh.stride(0) if has_biases else 0,
                    feature_size,
                    state_index,
                    direction,
                    dropout,
                    dropout_seed,
                    dropout_base,
                    HAS_BIAS=has_biases,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_H=64,
                    num_warps=4,
                    num_stages=2,
                )
        layer_outputs.append(layer_output)
        current_input = layer_output
        current_input_size = feature_size
        current_strides = layer_output.stride()

    if batch_first:
        output = _empty((batch_size, seq_len, directions * hidden_size), input)
        total = seq_len * batch_size * directions * hidden_size
        rnn_tanh_batch_first_copy_kernel[(triton.cdiv(total, 256),)](
            layer_outputs[-1],
            output,
            seq_len,
            batch_size,
            directions * hidden_size,
            BLOCK=256,
        )
    else:
        output = layer_outputs[-1]
    return output, hidden, layer_outputs


def _launch_backward(
    grad_output,
    grad_hidden,
    input,
    hx,
    params,
    layer_outputs,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
    dropout_seed,
    dropout_offset,
):
    if batch_first:
        batch_size, seq_len, first_input_size = input.shape
        first_input_strides = (input.stride(1), input.stride(0), input.stride(2))
        top_grad_strides = (
            grad_output.stride(1),
            grad_output.stride(0),
            grad_output.stride(2),
        )
    else:
        seq_len, batch_size, first_input_size = input.shape
        first_input_strides = input.stride()
        top_grad_strides = grad_output.stride()
    directions = 2 if bidirectional else 1
    hidden_size = hx.shape[2]
    rows = seq_len * batch_size
    grad_hx = _empty(hx.shape, hx)
    dh_read = _empty((batch_size, hidden_size), input)
    grad_params = [None] * len(params)
    # Autograd commonly supplies expanded zero-stride gradients for reductions
    # such as output.sum().  Materialize them with Triton so every downstream
    # kernel sees a regular time-major buffer without invoking aten::contiguous.
    top_feature_size = directions * hidden_size
    current_grad = _empty((seq_len, batch_size, top_feature_size), grad_output)
    grad_output_total = seq_len * batch_size * top_feature_size
    rnn_tanh_copy_3d_kernel[(triton.cdiv(grad_output_total, 256),)](
        grad_output,
        current_grad,
        seq_len,
        batch_size,
        top_feature_size,
        *top_grad_strides,
        BLOCK=256,
    )
    current_grad_strides = current_grad.stride()
    grad_hidden_contiguous = _empty(grad_hidden.shape, grad_hidden)
    grad_hidden_total = num_layers * directions * batch_size * hidden_size
    rnn_tanh_copy_3d_kernel[(triton.cdiv(grad_hidden_total, 256),)](
        grad_hidden,
        grad_hidden_contiguous,
        num_layers * directions,
        batch_size,
        hidden_size,
        *grad_hidden.stride(),
        BLOCK=256,
    )
    if runtime.device.vendor_name == "ascend":
        rnn_tanh_copy_3d_kernel[(triton.cdiv(grad_hidden_total, 256),)](
            grad_hidden_contiguous,
            grad_hx,
            num_layers * directions,
            batch_size,
            hidden_size,
            *grad_hidden_contiguous.stride(),
            BLOCK=256,
        )

    for layer in range(num_layers - 1, -1, -1):
        if layer == 0:
            layer_input = input
            input_size = first_input_size
            layer_input_strides = first_input_strides
            grad_layer_input = _empty(input.shape, input)
            grad_input_strides = first_input_strides
        else:
            layer_input = layer_outputs[layer - 1]
            input_size = directions * hidden_size
            layer_input_strides = layer_input.stride()
            grad_layer_input = _empty(layer_input.shape, layer_input)
            grad_input_strides = grad_layer_input.stride()

        apply_dropout = train and dropout > 0.0 and layer > 0
        dropout_base = dropout_offset + (layer - 1) * seq_len * batch_size * input_size
        for direction in range(directions):
            state_index = layer * directions + direction
            weight_ih, weight_hh, _, _ = _unpack_state_params(
                params, state_index, has_biases
            )
            dpre = _empty((seq_len, batch_size, hidden_size), input)
            if runtime.device.vendor_name == "ascend":
                rnn_tanh_bptt_vector_kernel[(batch_size,)](
                    current_grad,
                    grad_hidden_contiguous,
                    layer_outputs[layer],
                    weight_hh,
                    dpre,
                    grad_hx,
                    dh_read,
                    seq_len,
                    batch_size,
                    hidden_size,
                    *current_grad_strides,
                    directions * hidden_size,
                    *weight_hh.stride(),
                    state_index,
                    direction,
                    BLOCK_H=64,
                    num_warps=1,
                    num_stages=1,
                )
            elif hidden_size <= 256 and hidden_size >= 16:
                block_b = max(16, min(16, triton.next_power_of_2(batch_size)))
                block_h = max(16, triton.next_power_of_2(hidden_size))
                rnn_tanh_bptt_dot_kernel[(triton.cdiv(batch_size, block_b),)](
                    current_grad,
                    grad_hidden_contiguous,
                    layer_outputs[layer],
                    weight_hh,
                    dpre,
                    grad_hx,
                    seq_len,
                    batch_size,
                    hidden_size,
                    *current_grad_strides,
                    directions * hidden_size,
                    *weight_hh.stride(),
                    state_index,
                    direction,
                    BLOCK_B=block_b,
                    BLOCK_H=block_h,
                    num_warps=8 if block_h >= 128 else 4,
                    num_stages=2,
                )
            else:
                rnn_tanh_bptt_vector_kernel[(batch_size,)](
                    current_grad,
                    grad_hidden_contiguous,
                    layer_outputs[layer],
                    weight_hh,
                    dpre,
                    grad_hx,
                    dh_read,
                    seq_len,
                    batch_size,
                    hidden_size,
                    *current_grad_strides,
                    directions * hidden_size,
                    *weight_hh.stride(),
                    state_index,
                    direction,
                    BLOCK_H=64,
                    num_warps=4,
                )

            ascend_direction_grad = (
                _empty(grad_layer_input.shape, grad_layer_input)
                if runtime.device.vendor_name == "ascend" and direction != 0
                else grad_layer_input
            )
            if runtime.device.vendor_name == "ascend":
                rnn_tanh_dx_ascend_kernel[(rows, input_size)](
                    dpre,
                    weight_ih,
                    ascend_direction_grad,
                    rows,
                    batch_size,
                    input_size,
                    hidden_size,
                    *grad_input_strides,
                    *weight_ih.stride(),
                    dropout,
                    dropout_seed,
                    dropout_base,
                    ADD=False,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_H=64,
                )
                if direction != 0:
                    grad_layer_input_total = rows * input_size
                    rnn_tanh_add_kernel[(triton.cdiv(grad_layer_input_total, 256),)](
                        grad_layer_input,
                        ascend_direction_grad,
                        grad_layer_input_total,
                        BLOCK=256,
                    )
            else:
                rnn_tanh_dx_kernel[
                    (triton.cdiv(rows, 16), triton.cdiv(input_size, 32))
                ](
                    dpre,
                    weight_ih,
                    grad_layer_input,
                    rows,
                    batch_size,
                    input_size,
                    hidden_size,
                    *grad_input_strides,
                    *weight_ih.stride(),
                    dropout,
                    dropout_seed,
                    dropout_base,
                    ADD=direction != 0,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_M=16,
                    BLOCK_N=32,
                    BLOCK_K=32,
                )

            grad_weight_ih = _empty(weight_ih.shape, weight_ih)
            grad_weight_hh = _empty(weight_hh.shape, weight_hh)
            rnn_tanh_dw_ih_kernel[
                (triton.cdiv(hidden_size, 32), triton.cdiv(input_size, 32))
            ](
                dpre,
                layer_input,
                grad_weight_ih,
                rows,
                batch_size,
                input_size,
                hidden_size,
                *layer_input_strides,
                dropout,
                dropout_seed,
                dropout_base,
                APPLY_DROPOUT=apply_dropout,
                BLOCK_M=32,
                BLOCK_N=32,
                BLOCK_K=32,
            )
            if runtime.device.vendor_name == "ascend":
                rnn_tanh_dw_hh_ascend_kernel[(hidden_size, hidden_size)](
                    dpre,
                    layer_outputs[layer],
                    hx,
                    grad_weight_hh,
                    rows,
                    seq_len,
                    batch_size,
                    hidden_size,
                    directions * hidden_size,
                    *hx.stride(),
                    state_index,
                    direction,
                    BLOCK_K=64,
                )
            else:
                rnn_tanh_dw_hh_kernel[
                    (
                        triton.cdiv(hidden_size, 32),
                        triton.cdiv(hidden_size, 32),
                    )
                ](
                    dpre,
                    layer_outputs[layer],
                    hx,
                    grad_weight_hh,
                    rows,
                    seq_len,
                    batch_size,
                    hidden_size,
                    directions * hidden_size,
                    *hx.stride(),
                    state_index,
                    direction,
                    BLOCK_M=32,
                    BLOCK_N=32,
                    BLOCK_K=32,
                )
            width = 4 if has_biases else 2
            base = state_index * width
            grad_params[base] = grad_weight_ih
            grad_params[base + 1] = grad_weight_hh
            if has_biases:
                grad_bias_ih = _empty((hidden_size,), weight_ih)
                grad_bias_hh = _empty((hidden_size,), weight_ih)
                rnn_tanh_dbias_kernel[(triton.cdiv(hidden_size, 32),)](
                    dpre,
                    grad_bias_ih,
                    grad_bias_hh,
                    rows,
                    hidden_size,
                    BLOCK_H=32,
                    BLOCK_R=32,
                )
                grad_params[base + 2] = grad_bias_ih
                grad_params[base + 3] = grad_bias_hh

        current_grad = grad_layer_input
        current_grad_strides = grad_input_strides
    return current_grad, grad_hx, grad_params


def _packed_layout(batch_sizes, rows):
    # batch_sizes is intentionally CPU metadata in the ATen schema.  Reading it
    # here controls launch geometry only; all tensor math remains in Triton.
    sizes = tuple(int(value) for value in batch_sizes.tolist())
    if not sizes:
        raise RuntimeError("rnn_tanh.data: batch_sizes must not be empty")
    if sizes[0] <= 0 or any(
        current <= 0 or current > previous
        for previous, current in zip(sizes, sizes[1:])
    ):
        raise RuntimeError(
            "rnn_tanh.data: batch_sizes must be positive and non-increasing"
        )
    if sum(sizes) != rows:
        raise RuntimeError("rnn_tanh.data: sum(batch_sizes) must equal data.size(0)")
    offsets = []
    running = 0
    for size in sizes:
        offsets.append(running)
        running += size
    return sizes, tuple(offsets)


def _launch_packed_forward(
    data,
    batch_sizes_host,
    row_offsets,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    dropout_seed,
    dropout_offset,
):
    rows, first_input_size = data.shape
    max_batch = batch_sizes_host[0]
    directions = 2 if bidirectional else 1
    hidden_size = hx.shape[2]
    hidden = _empty((num_layers * directions, max_batch, hidden_size), data)
    hidden_total = num_layers * directions * max_batch * hidden_size
    rnn_tanh_copy_3d_kernel[(triton.cdiv(hidden_total, 256),)](
        hx,
        hidden,
        num_layers * directions,
        max_batch,
        hidden_size,
        *hx.stride(),
        BLOCK=256,
    )
    layer_outputs = []
    previous_states = []
    current_input = data
    current_input_size = first_input_size
    current_strides = data.stride()

    for layer in range(num_layers):
        feature_size = directions * hidden_size
        layer_output = _empty((rows, feature_size), data)
        apply_dropout = train and dropout > 0.0 and layer > 0
        dropout_base = dropout_offset + (layer - 1) * rows * current_input_size
        for direction in range(directions):
            state_index = layer * directions + direction
            weight_ih, weight_hh, bias_ih, bias_hh = _unpack_state_params(
                params, state_index, has_biases
            )
            previous = _empty((rows, hidden_size), data)
            previous_states.append(previous)
            use_dot = (
                current_input_size <= 256
                and hidden_size <= 256
                and current_input_size >= 16
                and hidden_size >= 16
                and data.dtype != torch.bfloat16
            )
            time_indices = (
                range(len(batch_sizes_host))
                if direction == 0
                else range(len(batch_sizes_host) - 1, -1, -1)
            )
            for time_index in time_indices:
                active_batch = batch_sizes_host[time_index]
                row_offset = row_offsets[time_index]
                if use_dot:
                    block_b = 16
                    block_i = max(16, triton.next_power_of_2(current_input_size))
                    block_h = max(16, triton.next_power_of_2(hidden_size))
                    rnn_tanh_packed_forward_step_dot_kernel[
                        (triton.cdiv(active_batch, block_b),)
                    ](
                        current_input,
                        hidden,
                        weight_ih,
                        weight_hh,
                        bias_ih,
                        bias_hh,
                        layer_output,
                        previous,
                        row_offset,
                        active_batch,
                        max_batch,
                        current_input_size,
                        hidden_size,
                        *current_strides,
                        *weight_ih.stride(),
                        *weight_hh.stride(),
                        bias_ih.stride(0) if has_biases else 0,
                        bias_hh.stride(0) if has_biases else 0,
                        feature_size,
                        state_index,
                        direction,
                        dropout,
                        dropout_seed,
                        dropout_base,
                        HAS_BIAS=has_biases,
                        APPLY_DROPOUT=apply_dropout,
                        BLOCK_B=block_b,
                        BLOCK_I=block_i,
                        BLOCK_H=block_h,
                        num_warps=8 if block_h >= 128 else 4,
                        num_stages=2,
                    )
                else:
                    rnn_tanh_packed_forward_step_vector_kernel[(active_batch,)](
                        current_input,
                        hidden,
                        weight_ih,
                        weight_hh,
                        bias_ih,
                        bias_hh,
                        layer_output,
                        previous,
                        row_offset,
                        active_batch,
                        max_batch,
                        current_input_size,
                        hidden_size,
                        *current_strides,
                        *weight_ih.stride(),
                        *weight_hh.stride(),
                        bias_ih.stride(0) if has_biases else 0,
                        bias_hh.stride(0) if has_biases else 0,
                        feature_size,
                        state_index,
                        direction,
                        dropout,
                        dropout_seed,
                        dropout_base,
                        HAS_BIAS=has_biases,
                        APPLY_DROPOUT=apply_dropout,
                        BLOCK_H=64,
                        num_warps=4,
                    )
        layer_outputs.append(layer_output)
        current_input = layer_output
        current_input_size = feature_size
        current_strides = layer_output.stride()
    return layer_outputs[-1], hidden, layer_outputs, previous_states


def _launch_packed_backward(
    grad_output,
    grad_hidden,
    data,
    batch_sizes_host,
    row_offsets,
    hx,
    params,
    layer_outputs,
    previous_states,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    dropout_seed,
    dropout_offset,
):
    rows, first_input_size = data.shape
    max_batch = batch_sizes_host[0]
    directions = 2 if bidirectional else 1
    hidden_size = hx.shape[2]
    feature_size = directions * hidden_size
    current_grad = _empty((rows, feature_size), grad_output)
    grad_output_total = rows * feature_size
    rnn_tanh_copy_3d_kernel[(triton.cdiv(grad_output_total, 256),)](
        grad_output,
        current_grad,
        rows,
        1,
        feature_size,
        grad_output.stride(0),
        0,
        grad_output.stride(1),
        BLOCK=256,
    )
    grad_hidden_work = _empty(grad_hidden.shape, grad_hidden)
    grad_hidden_total = num_layers * directions * max_batch * hidden_size
    rnn_tanh_copy_3d_kernel[(triton.cdiv(grad_hidden_total, 256),)](
        grad_hidden,
        grad_hidden_work,
        num_layers * directions,
        max_batch,
        hidden_size,
        *grad_hidden.stride(),
        BLOCK=256,
    )
    dh_read = _empty((max_batch, hidden_size), data)
    grad_params = [None] * len(params)

    for layer in range(num_layers - 1, -1, -1):
        if layer == 0:
            layer_input = data
            input_size = first_input_size
            layer_input_strides = data.stride()
        else:
            layer_input = layer_outputs[layer - 1]
            input_size = directions * hidden_size
            layer_input_strides = layer_input.stride()
        grad_layer_input = _empty((rows, input_size), layer_input)
        apply_dropout = train and dropout > 0.0 and layer > 0
        dropout_base = dropout_offset + (layer - 1) * rows * input_size
        for direction in range(directions):
            state_index = layer * directions + direction
            weight_ih, weight_hh, _, _ = _unpack_state_params(
                params, state_index, has_biases
            )
            dpre = _empty((rows, hidden_size), data)
            use_dot = (
                hidden_size <= 256
                and hidden_size >= 16
                and data.dtype != torch.bfloat16
            )
            time_indices = (
                range(len(batch_sizes_host) - 1, -1, -1)
                if direction == 0
                else range(len(batch_sizes_host))
            )
            for time_index in time_indices:
                active_batch = batch_sizes_host[time_index]
                row_offset = row_offsets[time_index]
                if runtime.device.vendor_name == "ascend":
                    rnn_tanh_bptt_step_kernel[
                        (active_batch, triton.cdiv(hidden_size, 64))
                    ](
                        current_grad,
                        grad_hidden_work,
                        layer_outputs[layer],
                        dpre,
                        row_offset,
                        active_batch,
                        max_batch,
                        hidden_size,
                        directions * hidden_size,
                        state_index,
                        direction,
                        BLOCK_H=64,
                    )
                    rnn_tanh_bptt_matvec_kernel[
                        (active_batch, triton.cdiv(hidden_size, 64))
                    ](
                        dpre,
                        weight_hh,
                        grad_hidden_work,
                        row_offset,
                        active_batch,
                        max_batch,
                        hidden_size,
                        *weight_hh.stride(),
                        state_index,
                        BLOCK_J=64,
                        BLOCK_K=32,
                    )
                elif use_dot:
                    block_b = 16
                    block_h = max(16, triton.next_power_of_2(hidden_size))
                    rnn_tanh_packed_bptt_step_dot_kernel[
                        (triton.cdiv(active_batch, block_b),)
                    ](
                        current_grad,
                        grad_hidden_work,
                        layer_outputs[layer],
                        weight_hh,
                        dpre,
                        row_offset,
                        active_batch,
                        max_batch,
                        hidden_size,
                        directions * hidden_size,
                        *weight_hh.stride(),
                        state_index,
                        direction,
                        BLOCK_B=block_b,
                        BLOCK_H=block_h,
                        num_warps=8 if block_h >= 128 else 4,
                    )
                else:
                    rnn_tanh_packed_bptt_step_vector_kernel[(active_batch,)](
                        current_grad,
                        grad_hidden_work,
                        layer_outputs[layer],
                        weight_hh,
                        dpre,
                        dh_read,
                        row_offset,
                        active_batch,
                        max_batch,
                        hidden_size,
                        directions * hidden_size,
                        *weight_hh.stride(),
                        state_index,
                        direction,
                        BLOCK_H=64,
                        num_warps=4,
                    )

            ascend_direction_grad = (
                _empty(grad_layer_input.shape, grad_layer_input)
                if runtime.device.vendor_name == "ascend" and direction != 0
                else grad_layer_input
            )
            if runtime.device.vendor_name == "ascend":
                rnn_tanh_dx_ascend_kernel[(rows, input_size)](
                    dpre,
                    weight_ih,
                    ascend_direction_grad,
                    rows,
                    1,
                    input_size,
                    hidden_size,
                    grad_layer_input.stride(0),
                    0,
                    grad_layer_input.stride(1),
                    *weight_ih.stride(),
                    dropout,
                    dropout_seed,
                    dropout_base,
                    ADD=False,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_H=64,
                )
                if direction != 0:
                    grad_layer_input_total = rows * input_size
                    rnn_tanh_add_kernel[(triton.cdiv(grad_layer_input_total, 256),)](
                        grad_layer_input,
                        ascend_direction_grad,
                        grad_layer_input_total,
                        BLOCK=256,
                    )
            else:
                rnn_tanh_packed_dx_kernel[
                    (triton.cdiv(rows, 16), triton.cdiv(input_size, 32))
                ](
                    dpre,
                    weight_ih,
                    grad_layer_input,
                    rows,
                    input_size,
                    hidden_size,
                    *weight_ih.stride(),
                    dropout,
                    dropout_seed,
                    dropout_base,
                    ADD=direction != 0,
                    APPLY_DROPOUT=apply_dropout,
                    BLOCK_M=16,
                    BLOCK_N=32,
                    BLOCK_K=32,
                )
            grad_weight_ih = _empty(weight_ih.shape, weight_ih)
            grad_weight_hh = _empty(weight_hh.shape, weight_hh)
            rnn_tanh_packed_dw_kernel[
                (triton.cdiv(hidden_size, 32), triton.cdiv(input_size, 32))
            ](
                dpre,
                layer_input,
                grad_weight_ih,
                rows,
                input_size,
                hidden_size,
                *layer_input_strides,
                dropout,
                dropout_seed,
                dropout_base,
                APPLY_DROPOUT=apply_dropout,
                BLOCK_M=32,
                BLOCK_N=32,
                BLOCK_K=32,
            )
            previous = previous_states[state_index]
            rnn_tanh_packed_dw_kernel[
                (triton.cdiv(hidden_size, 32), triton.cdiv(hidden_size, 32))
            ](
                dpre,
                previous,
                grad_weight_hh,
                rows,
                hidden_size,
                hidden_size,
                *previous.stride(),
                0.0,
                dropout_seed,
                0,
                APPLY_DROPOUT=False,
                BLOCK_M=32,
                BLOCK_N=32,
                BLOCK_K=32,
            )
            width = 4 if has_biases else 2
            base = state_index * width
            grad_params[base] = grad_weight_ih
            grad_params[base + 1] = grad_weight_hh
            if has_biases:
                grad_bias_ih = _empty((hidden_size,), weight_ih)
                grad_bias_hh = _empty((hidden_size,), weight_ih)
                rnn_tanh_dbias_kernel[(triton.cdiv(hidden_size, 32),)](
                    dpre,
                    grad_bias_ih,
                    grad_bias_hh,
                    rows,
                    hidden_size,
                    BLOCK_H=32,
                    BLOCK_R=32,
                )
                grad_params[base + 2] = grad_bias_ih
                grad_params[base + 3] = grad_bias_hh
        current_grad = grad_layer_input
    return current_grad, grad_hidden_work, grad_params


class RnnTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        hx,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        prefer_persistent_dot,
        *params,
    ):
        if batch_first:
            batch_size, seq_len = input.shape[:2]
        else:
            seq_len, batch_size = input.shape[:2]
        directions = 2 if bidirectional else 1
        random_values = (
            max(0, num_layers - 1) * seq_len * batch_size * directions * hx.shape[2]
        )
        dropout_seed, dropout_offset = _dropout_rng(
            random_values, train and dropout > 0.0
        )
        output, hidden, layer_outputs = _launch_forward(
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
            dropout_seed,
            dropout_offset,
            prefer_persistent_dot,
        )
        ctx.save_for_backward(input, hx, *params, *layer_outputs)
        ctx.param_count = len(params)
        ctx.has_biases = has_biases
        ctx.num_layers = num_layers
        ctx.dropout = dropout
        ctx.train = train
        ctx.bidirectional = bidirectional
        ctx.batch_first = batch_first
        ctx.dropout_seed = dropout_seed
        ctx.dropout_offset = dropout_offset
        return output, hidden

    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        saved = ctx.saved_tensors
        input, hx = saved[:2]
        params = saved[2 : 2 + ctx.param_count]
        layer_outputs = saved[2 + ctx.param_count :]
        grad_input, grad_hx, grad_params = _launch_backward(
            grad_output,
            grad_hidden,
            input,
            hx,
            params,
            layer_outputs,
            ctx.has_biases,
            ctx.num_layers,
            ctx.dropout,
            ctx.train,
            ctx.bidirectional,
            ctx.batch_first,
            ctx.dropout_seed,
            ctx.dropout_offset,
        )
        return (
            grad_input,
            grad_hx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grad_params,
        )


class RnnTanhDataFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data,
        batch_sizes,
        hx,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        *params,
    ):
        batch_sizes_host, row_offsets = _packed_layout(batch_sizes, data.shape[0])
        directions = 2 if bidirectional else 1
        random_values = (
            max(0, num_layers - 1) * data.shape[0] * directions * hx.shape[2]
        )
        dropout_seed, dropout_offset = _dropout_rng(
            random_values, train and dropout > 0.0
        )
        output, hidden, layer_outputs, previous_states = _launch_packed_forward(
            data,
            batch_sizes_host,
            row_offsets,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            dropout_seed,
            dropout_offset,
        )
        ctx.save_for_backward(data, hx, *params, *layer_outputs, *previous_states)
        ctx.param_count = len(params)
        ctx.num_layers = num_layers
        ctx.directions = 2 if bidirectional else 1
        ctx.batch_sizes_host = batch_sizes_host
        ctx.row_offsets = row_offsets
        ctx.has_biases = has_biases
        ctx.dropout = dropout
        ctx.train = train
        ctx.bidirectional = bidirectional
        ctx.dropout_seed = dropout_seed
        ctx.dropout_offset = dropout_offset
        return output, hidden

    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        saved = ctx.saved_tensors
        data, hx = saved[:2]
        param_end = 2 + ctx.param_count
        layer_end = param_end + ctx.num_layers
        params = saved[2:param_end]
        layer_outputs = saved[param_end:layer_end]
        previous_states = saved[layer_end:]
        grad_data, grad_hx, grad_params = _launch_packed_backward(
            grad_output,
            grad_hidden,
            data,
            ctx.batch_sizes_host,
            ctx.row_offsets,
            hx,
            params,
            layer_outputs,
            previous_states,
            ctx.has_biases,
            ctx.num_layers,
            ctx.dropout,
            ctx.train,
            ctx.bidirectional,
            ctx.dropout_seed,
            ctx.dropout_offset,
        )
        return (
            grad_data,
            None,
            grad_hx,
            None,
            None,
            None,
            None,
            None,
            *grad_params,
        )


def _rnn_tanh_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
    prefer_persistent_dot,
):
    params = tuple(params)
    _validate(input, hx, params, has_biases, num_layers, dropout, bidirectional)
    with runtime.torch_device_fn.device(input.device):
        return RnnTanhFunction.apply(
            input,
            hx,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
            prefer_persistent_dot,
            *params,
        )


def rnn_tanh(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """Apply a multi-layer Elman RNN with a tanh activation.

    This matches the tensor-input overload of ``torch.rnn_tanh`` and supports
    bias/no-bias, multiple layers, both directions, batch-first layout,
    training dropout, and pure-Triton backward.
    """
    logger.debug("GEMS RNN_TANH")
    prefer_persistent_dot = runtime.device.vendor_name in (
        "nvidia",
        "thead",
        "hygon",
    )
    return _rnn_tanh_impl(
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        prefer_persistent_dot=prefer_persistent_dot,
    )


def rnn_tanh_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    """PackedSequence overload of :func:`rnn_tanh`, implemented in Triton."""
    logger.debug("GEMS RNN_TANH DATA")
    if data.ndim != 2:
        raise RuntimeError(f"rnn_tanh.data: expected 2-D data, got {data.ndim}-D")
    params = tuple(params)
    directions = 2 if bidirectional else 1
    params_per_state = 4 if has_biases else 2
    expected_params = num_layers * directions * params_per_state
    if len(params) != expected_params:
        raise RuntimeError(
            f"rnn_tanh.data: expected {expected_params} parameters, "
            f"got {len(params)}"
        )
    if hx.ndim != 3 or hx.shape[0] != num_layers * directions:
        raise RuntimeError("rnn_tanh.data: hidden state has an invalid shape")
    if hx.shape[1] != int(batch_sizes[0]):
        raise RuntimeError("rnn_tanh.data: hidden batch size must match batch_sizes[0]")
    if dropout < 0.0 or dropout > 1.0:
        raise RuntimeError("rnn_tanh.data: dropout must be in [0, 1]")
    with runtime.torch_device_fn.device(data.device):
        return RnnTanhDataFunction.apply(
            data,
            batch_sizes,
            hx,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            *params,
        )
